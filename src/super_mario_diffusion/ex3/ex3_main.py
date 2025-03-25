import argparse
import itertools
import math
import os
import random
from typing import Dict, List
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler
)
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import matplotlib.pyplot as plt
from templates import imagenet_templates_small, imagenet_style_templates_small
from textual_inversion_dataset import TextualInversionDataset
from debug_utils import debug_dataloader

# Add these imports at the top of your file
import subprocess
import tempfile
import clip  

# And you'll need pytorch-fid: pip install pytorch-fid

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)
logger = get_logger(__name__, log_level="INFO")

CONCEPT_PROMPTS = [
    "A photo of {placeholder_token} playing soccer",
    "An action shot with lens flare and camera blur of {placeholder_token} waterskiing",
    "A cinematic photo of {placeholder_token} cooking while president barack obama is seated patiently",
    "A somber {placeholder_token} staring into a gorgeous lake in Patagonia"
]

CONFIG = {
    "pretrained_model": "stabilityai/stable-diffusion-2",
    "what_to_teach": "object",  # Choose between "object" or "style"
    "placeholder_token": "<my-concept>",  # The token you'll use to trigger your concept
    "initializer_token": "toy",  # A word that describes your concept
    # "learning_rate": 5e-04,
    "learning_rate": 7.5e-04,
    "scale_lr": True,  
    "max_train_steps": 1500,  # should be 2000
    "save_steps": 250,
    "train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "mixed_precision": "fp16",
    "seed": 42,
    "concept_folder": "example_input_concepts_lr2", # TODO: Change this to your concept folder,  sec 1.1 Concept Preparation
}
# Automatically set output_dir based on concept_folder
CONFIG["output_dir"] = "output_" + CONFIG["concept_folder"].rstrip("/") + "/"
os.makedirs(CONFIG["concept_folder"], exist_ok=True)
os.makedirs(CONFIG["output_dir"], exist_ok=True)

if not os.listdir(CONFIG["concept_folder"]):
    raise ValueError(
        f"The concept folder '{CONFIG['concept_folder']}' is empty! "
        "Please add 3-5 images of your concept before running the training."
    )


def image_grid(imgs: List[Image.Image], rows: int, cols: int) -> Image.Image:
    """Create a grid of images."""
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def setup_model_and_tokenizer(config: Dict) -> tuple:
    """Setup the model components and tokenizer."""
    tokenizer = CLIPTokenizer.from_pretrained(config["pretrained_model"], subfolder="tokenizer")
    
    # Add placeholder token
    num_added_tokens = tokenizer.add_tokens(config["placeholder_token"])
    if num_added_tokens == 0:
        raise ValueError(f"Token {config['placeholder_token']} already exists!")
        
    # Get token ids
    token_ids = tokenizer.encode(config["initializer_token"], add_special_tokens=False)
    if len(token_ids) > 1:
        raise ValueError("Initializer token must be a single token!")
        
    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(config["placeholder_token"])
    
    # Load models
    text_encoder = CLIPTextModel.from_pretrained(config["pretrained_model"], subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config["pretrained_model"], subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config["pretrained_model"], subfolder="unet")
    
    # Initialize placeholder token
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
    
    return tokenizer, text_encoder, vae, unet, placeholder_token_id

def freeze_models(text_encoder, vae, unet):
    """Freeze all parameters except the token embeddings."""
    def freeze_params(params):
        for param in params:
            param.requires_grad = False
            
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

def create_dataloader(batch_size, tokenizer):
    """Create the training dataloader."""
    train_dataset = TextualInversionDataset(
        data_root=CONFIG["concept_folder"],
        tokenizer=tokenizer,
        size=512,
        placeholder_token=CONFIG["placeholder_token"],
        repeats=100,
        learnable_property=CONFIG["what_to_teach"],
        center_crop_prob=0.5,  # 50% chance of center cropping
        flip_prob=0.5,  # 50% chance of flipping
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

def get_gpu_memory_info():
    """Get current and peak GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return 0, 0
    current = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
    peak = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
    return current, peak

def training_function(text_encoder, vae, unet, tokenizer, placeholder_token_id):
    # Check if MPS (Apple Silicon) is being used
    is_mps = torch.backends.mps.is_available()
    
    # Configure accelerator based on device
    accelerator = Accelerator(
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        mixed_precision='no' if is_mps else CONFIG["mixed_precision"],
    )

    train_batch_size = CONFIG["train_batch_size"]
    gradient_accumulation_steps = CONFIG["gradient_accumulation_steps"]
    learning_rate = CONFIG["learning_rate"]
    max_train_steps = CONFIG["max_train_steps"]
    output_dir = CONFIG["output_dir"]
    gradient_checkpointing = CONFIG["gradient_checkpointing"]

    # Initialize peak memory tracking
    peak_memory = 0
    
    if gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    train_dataloader = create_dataloader(train_batch_size, tokenizer)
    train_dataset = train_dataloader.dataset

    if CONFIG["scale_lr"]:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=learning_rate,
    )

    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # Keep vae in eval mode as we don't train it
    vae.eval()
    # Keep unet in train mode to enable gradient checkpointing
    unet.train()

    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(CONFIG["pretrained_model"], subfolder="scheduler")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    # Store original embeddings for token protection
    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()
    
    # Add learning rate scheduler for cosine decay
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=max_train_steps,
        eta_min=1e-6
    )

    # Track losses for visualization
    losses = []

    for epoch in range(num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Track GPU memory
            current_memory, peak = get_gpu_memory_info()
            peak_memory = max(peak_memory, peak)
            
            if global_step >= max_train_steps:
                break
                
            with accelerator.accumulate(text_encoder):
                # Convert images to latent space using VAE
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise to latents according to noise schedule
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

                # Predict noise with UNet
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Calculate loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                losses.append(loss.detach().item())
                
                accelerator.backward(loss)

                # Implement gradient clipping with ||∇||₂ ≤ 1.0 as required
                if accelerator.sync_gradients:
                    params_to_clip = text_encoder.get_input_embeddings().parameters()
                    accelerator.clip_grad_norm_(params_to_clip, max_norm=1.0)

                # Update parameters
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Ensure we don't update any embedding weights besides the placeholder token
                with torch.no_grad():
                    index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                    index_no_updates[placeholder_token_id] = False
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]

            # Save checkpoint every 100 steps as required
            if accelerator.is_main_process:
                if global_step % 100 == 0:
                    save_path = os.path.join(output_dir, f"learned_embeds-step-{global_step}.bin")
                    save_progress(text_encoder, placeholder_token_id, accelerator, save_path)
                    
                    # Plot and save loss curve at checkpoints
                    if len(losses) > 0:
                        plt.figure(figsize=(10, 5))
                        plt.plot(losses)
                        plt.title(f"Training Loss (Step {global_step})")
                        plt.xlabel("Step")
                        plt.ylabel("Loss")
                        plt.savefig(os.path.join(output_dir, f"loss_curve_step_{global_step}.png"))
                        plt.close()

                if global_step % 10 == 0:
                    logger.info(f"Step {global_step}: loss = {loss.detach().item():.4f}, GPU memory: {current_memory:.2f}GB, LR: {lr_scheduler.get_last_lr()[0]:.8f}")

            progress_bar.update(1)
            global_step += 1

        # Save the final checkpoint at the end of each epoch
        if accelerator.is_main_process:
            save_path = os.path.join(output_dir, f"learned_embeds-epoch-{epoch}.bin")
            save_progress(text_encoder, placeholder_token_id, accelerator, save_path)

    # Save the final loss curve
    if accelerator.is_main_process:
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(output_dir, "final_loss_curve.png"))
        plt.close()
        
    logger.info(f"Training completed. Peak GPU memory usage: {peak_memory:.2f}GB")

   ### TODO: Implement the training loop here for Section 1.2 Embedding Training
   ### 
   ### You need to:
   ### 1. Loop through epochs and batches
   ### 2. Process images through VAE to get latents
   ### 3. Add noise to latents using the noise scheduler
   ### 4. Get text embeddings from the text encoder
   ### 5. Predict noise with UNet and calculate loss
   ### 6. Update only the embeddings for the placeholder token
   ### 7. Save checkpoints at specified intervals
   ###
   ### Refer to the main.py file for implementation details
   # ...
   #########################################################

    
    logger.info(f"Training completed. Peak GPU memory usage: {peak_memory:.2f}GB")

def generate_concept_images(output_dir, pipeline):
    """Generate images using the trained concept with different guidance scales"""
    print("\n=== Testing Guidance Scales ===")
    
    # Test a range of guidance scales as required in the assignment
    guidance_scales = [7.5, 9.0, 11.0, 13.0, 15.0]
    
    # Use the global prompts
    test_prompts = [prompt.format(placeholder_token=CONFIG["placeholder_token"]) for prompt in CONCEPT_PROMPTS]
    
    # Create a grid with one row per prompt and one column per guidance scale
    # This makes it easy to compare how guidance scale affects each prompt
    all_images = []
    
    for prompt_idx, prompt in enumerate(test_prompts):
        prompt_images = []
        print(f"Generating images for prompt: '{prompt}'")
        
        for gs_idx, guidance_scale in enumerate(guidance_scales):
            print(f"  Using guidance scale: {guidance_scale}")
            
            # Use the same seed for each guidance scale to isolate the effect of guidance
            seed = prompt_idx * 100 + 42
            
            with torch.autocast("cuda"):
                image = pipeline(
                    prompt, 
                    num_inference_steps=40,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator("cuda").manual_seed(seed)
                ).images[0]
                
                prompt_images.append(image)
        
        all_images.extend(prompt_images)
    
    # Create and save the grid
    # Rows: Each prompt
    # Columns: Different guidance scales
    grid = image_grid(all_images, len(test_prompts), len(guidance_scales))
    grid_path = os.path.join(output_dir, "guidance_scale_grid.png")
    grid.save(grid_path)
    
    # Add guidance scale labels to the image
    labeled_grid = add_guidance_labels(grid, guidance_scales)
    labeled_path = os.path.join(output_dir, "labeled_guidance_scale_grid.png")
    labeled_grid.save(labeled_path)
    
    print(f"Guidance scale comparison grid saved to {grid_path}")
    return all_images

def add_guidance_labels(grid_image, guidance_scales):
    """Add guidance scale labels to the top of the grid image"""
    from PIL import ImageDraw, ImageFont
    
    # Make a copy to avoid modifying the original
    labeled_grid = grid_image.copy()
    draw = ImageDraw.Draw(labeled_grid)
    
    # Try to get a font, use default if not available
    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Get column width
    width, height = labeled_grid.size
    col_width = width // len(guidance_scales)
    
    # Add labels at the top of each column
    for i, gs in enumerate(guidance_scales):
        x = i * col_width + col_width // 2  # Center of column
        y = 10  # Top padding
        draw.text((x, y), f"GS: {gs}", fill="white", font=font, anchor="mt")  # mt = middle top
    
    return labeled_grid

def compare_embeddings(config_list):
    """Train and compare multiple embeddings with different configurations"""
    results = []
    
    for config_idx, config_params in enumerate(config_list):
        # Update configuration with new parameters
        for param, value in config_params.items():
            CONFIG[param] = value
        
        # Create a subdirectory for this configuration
        config_dir = os.path.join(CONFIG["output_dir"], f"config_{config_idx}")
        os.makedirs(config_dir, exist_ok=True)
        CONFIG["config_output_dir"] = config_dir
        
        print(f"\n=== Training Configuration {config_idx+1} ===")
        print(f"Parameters: {config_params}")
        
        # Setup models and tokenizer
        tokenizer, text_encoder, vae, unet, placeholder_token_id = setup_model_and_tokenizer(CONFIG)
        freeze_models(text_encoder, vae, unet)
        
        # Train with this configuration
        training_function(text_encoder, vae, unet, tokenizer, placeholder_token_id)
        
        # Save the model
        pipeline = StableDiffusionPipeline.from_pretrained(
            CONFIG["pretrained_model"],
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(config_dir)
        
        # Generate sample images
        pipeline = pipeline.to("cuda")
        generated_images = []
        
        # Generate images using all prompts
        prompts = [prompt.format(placeholder_token=CONFIG["placeholder_token"]) for prompt in CONCEPT_PROMPTS]
        for prompt_idx, prompt in enumerate(prompts):
            for seed_idx in range(3):  # Generate 3 samples per prompt
                with torch.autocast("cuda"):
                    image = pipeline(
                        prompt,
                        num_inference_steps=40,
                        guidance_scale=7.5,
                        generator=torch.Generator("cuda").manual_seed(prompt_idx*10 + seed_idx)
                    ).images[0]
                    generated_images.append(image)
        
        # Save grid of generated images
        grid = image_grid(generated_images, len(prompts), 3)
        grid.save(os.path.join(config_dir, "generated_grid.png"))
        
        # Calculate metrics compared to concept images
        metrics = compute_metrics(CONFIG["concept_folder"], generated_images)
        
        # Store results
        results.append({
            "config": config_params,
            "embedding": text_encoder.get_input_embeddings().weight[placeholder_token_id].detach().cpu(),
            "output_dir": config_dir,
            "metrics": metrics
        })
        
        print(f"Configuration {config_idx+1} results:")
        print(f"  FID score: {metrics['fid']:.4f}")
        print(f"  CLIP score: {metrics['clip_score']:.4f}")
        
        # Clean up
        del pipeline, text_encoder, vae, unet
        torch.cuda.empty_cache()
    
    # Compare embeddings and metrics
    if len(results) > 1:
        print("\n=== Configuration Comparison ===")
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                emb_i = results[i]["embedding"]
                emb_j = results[j]["embedding"]
                
                # Calculate cosine similarity between embeddings
                cos_sim = F.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0)).item()
                
                # Calculate L2 distance between embeddings
                l2_dist = torch.norm(emb_i - emb_j, p=2).item()
                
                print(f"Config {i+1} vs Config {j+1}:")
                print(f"  Cosine similarity: {cos_sim:.4f}")
                print(f"  L2 distance: {l2_dist:.4f}")
                print(f"  FID difference: {abs(results[i]['metrics']['fid'] - results[j]['metrics']['fid']):.4f}")
                print(f"  CLIP score difference: {abs(results[i]['metrics']['clip_score'] - results[j]['metrics']['clip_score']):.4f}")
    
    # Create a comparison summary table
    print("\n=== Metrics Summary ===")
    headers = ["Config", "Learning Rate", "FID Score", "CLIP Score"]
    rows = []
    for i, result in enumerate(results):
        rows.append([
            f"Config {i+1}",
            f"{result['config'].get('learning_rate', 'N/A')}",
            f"{result['metrics']['fid']:.4f}",
            f"{result['metrics']['clip_score']:.4f}"
        ])
    
    # Print as markdown table
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---" for _ in headers]) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")

    # Save metrics to a file
    metrics_path = os.path.join(CONFIG["output_dir"], "metrics_summary.txt")
    with open(metrics_path, 'w') as metrics_file:
        # Write a header
        metrics_file.write("=== Textual Inversion Training Metrics ===\n\n")
        metrics_file.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        metrics_file.write(f"Concept: {CONFIG['placeholder_token']}\n\n")
        
        # Write the table
        metrics_file.write(table_str)
        metrics_file.write("\n\n")
        
        # Write detailed comparison
        if len(results) > 1:
            metrics_file.write("=== Configuration Comparison ===\n\n")
            for i in range(len(results)):
                for j in range(i+1, len(results)):
                    emb_i = results[i]["embedding"]
                    emb_j = results[j]["embedding"]
                    
                    # Calculate cosine similarity between embeddings
                    cos_sim = F.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0)).item()
                    
                    # Calculate L2 distance between embeddings
                    l2_dist = torch.norm(emb_i - emb_j, p=2).item()
                    
                    metrics_file.write(f"Config {i+1} vs Config {j+1}:\n")
                    metrics_file.write(f"  Cosine similarity: {cos_sim:.4f}\n")
                    metrics_file.write(f"  L2 distance: {l2_dist:.4f}\n")
                    metrics_file.write(f"  FID difference: {abs(results[i]['metrics']['fid'] - results[j]['metrics']['fid']):.4f}\n")
                    metrics_file.write(f"  CLIP score difference: {abs(results[i]['metrics']['clip_score'] - results[j]['metrics']['clip_score']):.4f}\n\n")
        
        # Write individual configuration details
        metrics_file.write("=== Individual Configuration Details ===\n\n")
        for i, result in enumerate(results):
            metrics_file.write(f"Config {i+1}:\n")
            for param, value in result["config"].items():
                metrics_file.write(f"  {param}: {value}\n")
            metrics_file.write(f"  FID score: {result['metrics']['fid']:.4f}\n")
            metrics_file.write(f"  CLIP score: {result['metrics']['clip_score']:.4f}\n\n")
    
    print(f"Metrics summary saved to {metrics_path}")
    
    return results

def save_progress(text_encoder, placeholder_token_id, accelerator, save_path):
    """Helper function to save the trained embeddings."""
    logger = get_logger(__name__)
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {CONFIG["placeholder_token"]: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)

# def compute_metrics(real_images_dir, generated_images, model_name="ViT-B/32"):
#     """Compute FID and CLIP scores between real images and generated images"""
#     print("Computing FID and CLIP scores...")
    
#     # Create a temporary directory to store generated images for FID calculation
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         # Save generated images to temp directory
#         for i, img in enumerate(generated_images):
#             img.save(os.path.join(tmp_dir, f"gen_{i:04d}.png"))
        
#         # Calculate FID score
#         try:
#             # Using pytorch-fid
#             fid_value = float(subprocess.check_output([
#                 'python', '-m', 'pytorch_fid', 
#                 real_images_dir, 
#                 tmp_dir
#             ]).decode('utf-8').strip().split()[-1])
#         except (subprocess.SubprocessError, ValueError) as e:
#             print(f"Error calculating FID score: {e}")
#             fid_value = float('nan')
    
#     # Load CLIP model
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     clip_model, preprocess = clip.load(model_name, device=device)
    
#     # Process real images with CLIP
#     real_images = []
#     for filename in os.listdir(real_images_dir):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
#             try:
#                 image_path = os.path.join(real_images_dir, filename)
#                 image = Image.open(image_path).convert('RGB')
#                 real_images.append(preprocess(image).unsqueeze(0).to(device))
#             except Exception as e:
#                 print(f"Error loading image {image_path}: {e}")
    
#     if not real_images:
#         print("No valid real images found.")
#         return {'fid': fid_value, 'clip_score': float('nan')}
    
#     real_batch = torch.cat(real_images)
    
#     # Process generated images with CLIP
#     gen_images = [preprocess(img).unsqueeze(0).to(device) for img in generated_images]
#     gen_batch = torch.cat(gen_images)
    
#     # Calculate CLIP similarity
#     with torch.no_grad():
#         real_features = clip_model.encode_image(real_batch)
#         gen_features = clip_model.encode_image(gen_batch)
        
#         # Normalize features
#         real_features = real_features / real_features.norm(dim=1, keepdim=True)
#         gen_features = gen_features / gen_features.norm(dim=1, keepdim=True)
        
#         # Calculate mean cosine similarity between each generated image and all real images
#         similarity = torch.mm(gen_features, real_features.T)
#         clip_score = similarity.mean().item()
    
#     return {'fid': fid_value, 'clip_score': clip_score}

def compute_metrics(real_images_dir, generated_images, model_name="ViT-B/32"):
    """Compute FID and CLIP scores between real images and generated images"""
    print("Computing FID and CLIP scores...")
    
    # Free GPU memory before metrics computation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    import gc
    gc.collect()
    
    # Calculate FID score
    fid_value = float('nan')  # Default value if calculation fails
    try:
        # Create a temporary directory for generated images
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            print(f"Saving generated images to temporary directory: {tmp_dir}")
            # Save generated images for FID calculation
            for i, img in enumerate(generated_images):
                img.save(os.path.join(tmp_dir, f"gen_{i:04d}.png"))
            
            print(f"Saved {len(generated_images)} generated images")
            print(f"Starting FID calculation...")
            
            try:
                # Run pytorch-fid as subprocess
                import subprocess
                result = subprocess.run(
                    ['python', '-m', 'pytorch_fid', real_images_dir, tmp_dir],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"FID calculation output: {result.stdout}")
                fid_value = float(result.stdout.strip().split()[-1])
                print(f"FID score: {fid_value}")
            except Exception as e:
                print(f"FID calculation failed: {str(e)}")
                if hasattr(e, 'stderr'):
                    print(f"Error output: {e.stderr}")
    except Exception as e:
        print(f"Error preparing for FID calculation: {str(e)}")
    
    # Calculate CLIP score - using CPU to avoid CUDA issues
    clip_score = float('nan')
    try:
        print("Loading CLIP model for CLIP score calculation...")
        # Ensure we use CPU for CLIP
        device = "cpu"
        
        # Import clip properly
        try:
            import clip
        except ImportError:
            print("CLIP not found, installing...")
            os.system("pip install git+https://github.com/openai/CLIP.git")
            import clip
        
        model, preprocess = clip.load(model_name, device=device)
        
        # Process real images
        real_features = []
        print(f"Processing real images from: {real_images_dir}")
        for filename in os.listdir(real_images_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                try:
                    image_path = os.path.join(real_images_dir, filename)
                    image = Image.open(image_path).convert('RGB')
                    image_input = preprocess(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feature = model.encode_image(image_input)
                        # Normalize feature
                        feature = feature / feature.norm(dim=1, keepdim=True)
                        real_features.append(feature)
                except Exception as e:
                    print(f"Error processing real image {filename}: {str(e)}")
        
        if not real_features:
            print("No valid real images found!")
            return {'fid': fid_value, 'clip_score': float('nan')}
        
        # Combine all real features
        real_features = torch.cat(real_features, dim=0)
        
        # Process generated images
        gen_features = []
        print("Processing generated images...")
        for i, img in enumerate(generated_images):
            try:
                image_input = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feature = model.encode_image(image_input)
                    # Normalize feature
                    feature = feature / feature.norm(dim=1, keepdim=True)
                    gen_features.append(feature)
            except Exception as e:
                print(f"Error processing generated image {i}: {str(e)}")
        
        # Combine all generated features
        gen_features = torch.cat(gen_features, dim=0)
        
        # Calculate similarity
        print("Calculating CLIP similarity...")
        with torch.no_grad():
            similarity = torch.mm(gen_features, real_features.T)
            clip_score = similarity.mean().item()
            print(f"CLIP score: {clip_score}")
        
    except Exception as e:
        print(f"Error calculating CLIP score: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return {'fid': fid_value, 'clip_score': clip_score}


# def main():
#     print(f"Starting textual inversion training...")
#     print(f"Using concept images from: {CONFIG['concept_folder']}")
#     print(f"Number of concept images: {len(os.listdir(CONFIG['concept_folder']))}")
    
#     # Set seed for reproducibility
#     set_seed(CONFIG["seed"])
    
#     # Setup
#     tokenizer, text_encoder, vae, unet, placeholder_token_id = setup_model_and_tokenizer(CONFIG)
    
#     # Debug dataloader before training
#     debug_dataloader(tokenizer, CONFIG)
    
#     # Continue with training
#     freeze_models(text_encoder, vae, unet)
    
#     # Train
#     training_function(text_encoder, vae, unet, tokenizer, placeholder_token_id)
    
#     # Save the final model
#     pipeline = StableDiffusionPipeline.from_pretrained(
#         CONFIG["pretrained_model"],
#         text_encoder=text_encoder,
#         tokenizer=tokenizer,
#         vae=vae,
#         unet=unet,
#     )
#     pipeline.save_pretrained(CONFIG["output_dir"])
#     print(f"Training completed. Model saved to {CONFIG['output_dir']}")

#     # Copy concept folder images as a grid in the output folder
#     print("Creating a grid of concept images...")
#     concept_images = []
#     for image_file in os.listdir(CONFIG["concept_folder"]):
#         if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
#             image_path = os.path.join(CONFIG["concept_folder"], image_file)
#             try:
#                 img = Image.open(image_path).convert('RGB')
#                 concept_images.append(img)
#             except Exception as e:
#                 print(f"Error loading image {image_path}: {e}")
    
#     if concept_images:
#         # Calculate grid dimensions
#         num_images = len(concept_images)
#         cols = min(4, num_images)  # Max 4 columns
#         rows = math.ceil(num_images / cols)
        
#         # Pad with blank images if needed
#         while len(concept_images) < rows * cols:
#             blank = Image.new('RGB', concept_images[0].size, color=(255, 255, 255))
#             concept_images.append(blank)
        
#         # Create and save the grid
#         concept_grid = image_grid(concept_images, rows, cols)
#         concept_grid_path = os.path.join(CONFIG["output_dir"], "concelo que quieras. Te daré lo quept_images_grid.png")
#         concept_grid.save(concept_grid_path)
#         print(f"Concept images grid saved to {concept_grid_path}")
#     else:
#         print("No valid images found in the concept folder to create a grid.")


#     embedding_configs = [
#         # {"learning_rate": 5e-04, "initializer_token": "toy"},
#         # {"learning_rate": 1e-03, "initializer_token": "toy"},
#         # {"learning_rate": 5e-04, "initializer_token": "person"}
#         {"learning_rate": 7.5e-04, "initializer_token": "toy"},
#         {"learning_rate": 1e-03, "initializer_token": "toy", "max_train_steps": 1000},


#     ]
    
#     results = compare_embeddings(embedding_configs)
    
#    # 1.3 Concept Generation
#     print("\n=== 1.3 Concept Generation ===")
#     print("Generating example images with the trained concept...")

#     # 1. Load the trained pipeline from the output directory
#     print(f"Loading trained pipeline from {CONFIG['output_dir']}...")
#     trained_pipeline = StableDiffusionPipeline.from_pretrained(
#         CONFIG["output_dir"],
#         torch_dtype=torch.float16,
#     )

#     # 2. Configure the DPMSolverMultistepScheduler for efficient sampling
#     trained_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
#         trained_pipeline.scheduler.config,
#         algorithm_type="dpmsolver++",
#         solver_order=2
#     )

#     # 3. Move the model to GPU
#     trained_pipeline = trained_pipeline.to("cuda")

#     # 4. Generate images with different guidance scales
#     generated_images = generate_concept_images(CONFIG["output_dir"], trained_pipeline)

#     # The rest of your generation code...

#     # 5. Generate a grid of base concept images (2 per prompt)
#     print("Generating base concept images...")
#     generated_images = []
#     for i, prompt in enumerate(prompts):
#         print(f"  Generating for prompt {i+1}/{len(prompts)}: '{prompt}'")
#         # Generate two samples for each prompt
#         for j in range(2):
#             with torch.autocast("cuda"):
#                 image = trained_pipeline(
#                     prompt, 
#                     num_inference_steps=40,
#                     guidance_scale=7.5,
#                     generator=torch.Generator("cuda").manual_seed(i*10 + j)
#                 ).images[0]
#                 generated_images.append(image)

#     # 6. Create and save the grid
#     grid = image_grid(generated_images, len(prompts), 2)  # 2 samples per prompt
#     grid_path = os.path.join(CONFIG["output_dir"], "generated_concept_grid.png")
#     grid.save(grid_path)
#     print(f"Base concept grid saved to {grid_path}")
#     # Save individual images too
#     sample_dir = os.path.join(CONFIG["output_dir"], "generated_samples")
#     os.makedirs(sample_dir, exist_ok=True)
#     for i, image in enumerate(generated_images):
#         image_path = os.path.join(sample_dir, f"generated_{i}.png")
#         image.save(image_path)

#     # Optional: Test different prompt templates
#     print("\nGenerating images with different contexts...")
#     context_modifiers = [
#         "in the style of Van Gogh",
#         "made of gold",
#         "in a sci-fi setting", 
#         "looking happy"
#     ]

#     context_images = []
#     context_prompts = []
#     for i, prompt in enumerate(CONCEPT_PROMPTS):
#         # Take the prompt and add the modifier
#         modified = prompt.format(placeholder_token=CONFIG["placeholder_token"])
#         # Insert the modifier before the last word
#         words = modified.split()
#         modified_prompt = " ".join(words[:-1]) + f" {context_modifiers[i%len(context_modifiers)]} " + words[-1]
#         context_prompts.append(modified_prompt)
        
#         # Generate the image directly and add to list for grid
#         print(f"  Generating context variation {i+1}/{len(CONCEPT_PROMPTS)}: '{modified_prompt}'")
#         with torch.autocast("cuda"):
#             image = trained_pipeline(
#                 modified_prompt,
#                 num_inference_steps=40,
#                 guidance_scale=7.5,
#                 generator=torch.Generator("cuda").manual_seed(42 + i)
#             ).images[0]
#             context_images.append(image)

#     # Create and save context grid
#     context_grid = image_grid(context_images, 2, 2)
#     context_grid_path = os.path.join(CONFIG["output_dir"], "generated_context_grid.png")
#     context_grid.save(context_grid_path)
#     print(f"Context variations grid saved to {context_grid_path}")

#     # Clean up to save memory
#     del trained_pipeline
#     torch.cuda.empty_cache()
#     print("Concept generation complete.")
def main():
    print(f"Starting textual inversion training...")
    print(f"Using concept images from: {CONFIG['concept_folder']}")
    print(f"Number of concept images: {len(os.listdir(CONFIG['concept_folder']))}")
    
    # Set seed for reproducibility
    set_seed(CONFIG["seed"])
    
    # Setup
    tokenizer, text_encoder, vae, unet, placeholder_token_id = setup_model_and_tokenizer(CONFIG)
    
    # Debug dataloader before training
    debug_dataloader(tokenizer, CONFIG)
    
    # Continue with training
    freeze_models(text_encoder, vae, unet)
    
    # Train just once with the default configuration
    print("\n=== Starting Training ===")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print(f"Initializer token: {CONFIG['initializer_token']}")
    print(f"Max training steps: {CONFIG['max_train_steps']}")
    
    # Free GPU memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Train
    training_function(text_encoder, vae, unet, tokenizer, placeholder_token_id)
    
    # Save the final model
    pipeline = StableDiffusionPipeline.from_pretrained(
        CONFIG["pretrained_model"],
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        unet=unet,
    )
    pipeline.save_pretrained(CONFIG["output_dir"])
    print(f"Training completed. Model saved to {CONFIG['output_dir']}")
    
    # Now generate images for evaluation
    print("\n=== Generating Images for Evaluation ===")
    
    # Move the model to GPU for generation
    pipeline = pipeline.to("cuda")
    generated_images = []
    
    # Generate images using prompts
    prompts = [prompt.format(placeholder_token=CONFIG["placeholder_token"]) for prompt in CONCEPT_PROMPTS]
    for prompt_idx, prompt in enumerate(prompts):
        print(f"Generating images for prompt: '{prompt}'")
        for seed_idx in range(3):  # Generate 3 samples per prompt
            with torch.autocast("cuda"):
                image = pipeline(
                    prompt,
                    num_inference_steps=40,
                    guidance_scale=7.5,
                    generator=torch.Generator("cuda").manual_seed(prompt_idx*10 + seed_idx)
                ).images[0]
                generated_images.append(image)
    
    # Save grid of generated images
    grid = image_grid(generated_images, len(prompts), 3)
    grid.save(os.path.join(CONFIG["output_dir"], "generated_grid.png"))
    print(f"Generated image grid saved to {CONFIG['output_dir']}/generated_grid.png")
    
    # Calculate FID and CLIP scores
    print("\n=== Calculating FID and CLIP Scores ===")
    # Make sure we free GPU memory before computing metrics
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    import gc
    gc.collect()
    
    # Use the fixed compute_metrics function to calculate scores
    try:
        metrics = compute_metrics(CONFIG["concept_folder"], generated_images)
        
        print(f"Evaluation metrics:")
        print(f"  FID score: {metrics['fid']:.4f}")
        print(f"  CLIP score: {metrics['clip_score']:.4f}")
        
        # Save metrics to a file
        metrics_path = os.path.join(CONFIG["output_dir"], "metrics.txt")
        with open(metrics_path, 'w') as metrics_file:
            metrics_file.write(f"FID score: {metrics['fid']:.4f}\n")
            metrics_file.write(f"CLIP score: {metrics['clip_score']:.4f}\n")
        
        print(f"Metrics saved to {metrics_path}")
    except Exception as e:
        print(f"Error calculating metrics: {e}")
    
    print("\nTextual inversion training and evaluation completed.")






if __name__ == "__main__":
    main()
