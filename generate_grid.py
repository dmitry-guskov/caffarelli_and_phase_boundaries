# generate_latent_slice_grid.py
import os, math, numpy as np, torch
from PIL import Image
from tqdm import tqdm

from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
class WanPipelineWithSnapshots(WanPipeline):
    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            negative_prompt: Union[str, List[str]] = None,
            height: int = 480,
            width: int = 832,
            num_frames: int = 81,
            num_inference_steps: int = 50,
            guidance_scale: float = 5.0,
            guidance_scale_2: Optional[float] = None,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            output_type: Optional[str] = "np",
            return_dict: bool = True,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            max_sequence_length: int = 512,
            capture_idxs = [0, 5, 10, 15, 20, 25]
        ):
            if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
                callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
    
            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                negative_prompt,
                height,
                width,
                prompt_embeds,
                negative_prompt_embeds,
                callback_on_step_end_tensor_inputs,
                guidance_scale_2,
            )
            snapshots = {}
            if num_frames % self.vae_scale_factor_temporal != 1:
                logger.warning(
                    f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
                )
                num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
            num_frames = max(num_frames, 1)
    
            if self.config.boundary_ratio is not None and guidance_scale_2 is None:
                guidance_scale_2 = guidance_scale
    
            self._guidance_scale = guidance_scale
            self._guidance_scale_2 = guidance_scale_2
            self._attention_kwargs = attention_kwargs
            self._current_timestep = None
            self._interrupt = False
    
            device = self._execution_device
    
            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]
    
            # 3. Encode input prompt
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                device=device,
            )
    
            transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
            prompt_embeds = prompt_embeds.to(transformer_dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
    
            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
    
            # 5. Prepare latent variables
            num_channels_latents = (
                self.transformer.config.in_channels
                if self.transformer is not None
                else self.transformer_2.config.in_channels
            )
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                num_frames,
                torch.float32,
                device,
                generator,
                latents,
            )
    
            mask = torch.ones(latents.shape, dtype=torch.float32, device=device)
    
            # 6. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)
    
            if self.config.boundary_ratio is not None:
                boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
            else:
                boundary_timestep = None
    
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue
    
                    self._current_timestep = t
    
                    if boundary_timestep is None or t >= boundary_timestep:
                        # wan2.1 or high-noise stage in wan2.2
                        current_model = self.transformer
                        current_guidance_scale = guidance_scale
                    else:
                        # low-noise stage in wan2.2
                        current_model = self.transformer_2
                        current_guidance_scale = guidance_scale_2
    
                    latent_model_input = latents.to(transformer_dtype)
                    if self.config.expand_timesteps:
                        # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                        temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                        # batch_size, seq_len
                        timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                    else:
                        timestep = t.expand(latents.shape[0])
    
                    with current_model.cache_context("cond"):
                        noise_pred = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
    
                    if self.do_classifier_free_guidance:
                        with current_model.cache_context("uncond"):
                            noise_uncond = current_model(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)
                    ####
                    x_t = latents  
                    # 1) Try sigma-based (Euler/UniPC/FlowMatch) x_t = (1-σ) x0 + σ ε  =>  x0 = (x_t - σ ε)/(1-σ)
                    pred_x0 = None
                    if hasattr(pipe.scheduler, "sigmas"):
                        # robust index from timestep tensor t
                        idx = pipe.scheduler.index_for_timestep(t, pipe.scheduler.timesteps)
                        sigma = pipe.scheduler.sigmas[idx].to(x_t.device, x_t.dtype)  # scalar
                        pred_x0 = (x_t - sigma * noise_pred) / (1.0 - sigma + 1e-8)
                    
                    # 2) Fallback: DDIM/DDPM ᾱt formula (if alphas_cumprod exists)
                    elif hasattr(pipe.scheduler, "alphas_cumprod"):
                        t_idx = int(t.flatten()[0].item())
                        a_bar = pipe.scheduler.alphas_cumprod.to(x_t.device, x_t.dtype)[t_idx]  # scalar
                        sa, s1m = a_bar.sqrt(), (1 - a_bar).sqrt()
                        pred_type = getattr(getattr(pipe.scheduler, "config", None), "prediction_type", "epsilon")
                        if pred_type in ("epsilon", "eps"):
                            pred_x0 = (x_t - s1m * noise_pred) / (sa + 1e-8)
                        elif pred_type in ("v_prediction", "v-prediction", "v"):
                            pred_x0 = sa * x_t - s1m * noise_pred
                        elif pred_type in ("sample", "x0"):
                            pred_x0 = noise_pred
                    snapshots[i] = pred_x0
                    ####


                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
    
                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
    
                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
    
                    # if XLA_AVAILABLE:
                    #     xm.mark_step()
    
            self._current_timestep = None
    
            if not output_type == "latent":
                latents = latents.to(self.vae.dtype)
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                    latents.device, latents.dtype
                )
                latents = latents / latents_std + latents_mean
                video = self.vae.decode(latents, return_dict=False)[0]
                video = self.video_processor.postprocess_video(video, output_type=output_type)
            else:
                video = latents
    
            # Offload all models
            self.maybe_free_model_hooks()
    
            # if not return_dict:
            #     return (video,)
    
            # return WanPipelineOutput(frames=video)
            if not return_dict:
                return (video, snapshots)
            return WanPipelineOutput(frames=video), snapshots
    @torch.no_grad()    
    def convert_latent(self, latents,
                      output_type: Optional[str] = "np",):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return WanPipelineOutput(frames=video)

        
def interpolated_latent(latent_0, latent_1, latent_2, alpha, beta, noise=1e-6):
    '''
    interpoolates gaussians 
    '''
    interpolated_latent =  latent_0  + alpha * (latent_1 - latent_0) + beta * (latent_2 - latent_0) + torch.randn_like(latent_0) * noise
    return interpolated_latent / interpolated_latent.std()

# -------------------- Config --------------------
OUT_ROOT        = "./data/wan21_slice"
STEP_DIRS       = {0: "step_000", 5: "step_005", 10: "step_010", 15: "step_015", 49: "step_049"}
MODEL_ID        = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODELS_ROOT     = "./models"
PROMPT          =  'High quality picture, 4k, detailed'
NEG_PROMPT      = 'blurry, ugly, stock photo'
HEIGHT, WIDTH   = 480, 480
NUM_STEPS       = 50      # 0..49
GUIDANCE        = 5.0
NUM_FRAMES      = 1       # image
GRID_N          = 100


# Make output folders
for step_idx, sub in STEP_DIRS.items():
    os.makedirs(os.path.join(OUT_ROOT, sub, "images"), exist_ok=True)

# -------------------- Load pipeline --------------------
device = "cuda:3" if torch.cuda.is_available() else "cpu"
pipe_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

# If you use a subclass with snapshots:
from diffusers import WanPipeline, AutoencoderKLWan
vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae",
                                       torch_dtype=torch.float32,
                                       cache_dir=MODELS_ROOT,
                                       low_cpu_mem_usage=True)
# If you subclassed: WanPipelineWithSnapshots
pipe = WanPipelineWithSnapshots.from_pretrained(MODEL_ID, vae=vae, torch_dtype=pipe_dtype,
                                                 cache_dir=MODELS_ROOT, low_cpu_mem_usage=True).to(device)

from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

# -------------------- Build latent anchors --------------------
latents = np.load('latents_1.npy')
z0,z1,z2 = [torch.tensor(i) for i in latents]
# -------------------- Grid --------------------
alpha_values = np.linspace(0.0, 1.0, GRID_N, dtype=np.float64)
beta_values  = np.linspace(0.0, 1.0, GRID_N, dtype=np.float64)

def save_img(np_img: np.ndarray, out_dir: str, hash_id: str):
    img = Image.fromarray(np_img.astype("uint8"), mode="RGB")
    img.save(os.path.join(out_dir, "images", f"x0_{hash_id}.png"))
    # optional: sidecar
    np.save(os.path.join(out_dir, "images", f"x0_{hash_id}.npy"),
            {"image": np_img[None, ...]}, allow_pickle=True)

def decode_final_video_to_np(video) -> np.ndarray:
    # Wan returns np arrays already (H, W, 3) for one frame; handle list variants.
    frame = video
    if isinstance(frame, list):
        frame = frame[0]
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    return frame

for i, alpha in enumerate(tqdm(alpha_values, desc="alpha")):
    for j, beta in enumerate(beta_values):
        # weights for z0,z1,z2
        init_latents = interpolated_latent(z0, z1, z2, alpha, beta,noise=0)

        # Call pipeline. We expect (final_video, snapshots) when return_dict=False
        out = pipe(
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE,
            latents=init_latents,
            output_type="np",
            return_dict=False,
        )
        # Expect either (video, snapshots) or just (video,)
        if isinstance(out, tuple) and len(out) == 2:
            final_video, snapshots = out
        else:
            final_video = out[0]
            snapshots = {}

        # Naming
        hash_id = f"a{alpha_values[i]:.4f}_b{beta_values[j]:.4f}_i{i:03d}_j{j:03d}"

        # Save requested intermediate x0 frames if available
        for step_idx in (0, 5, 10, 15):
            if step_idx in snapshots:
                np_img = decode_final_video_to_np(snapshots[step_idx])
                save_img(np_img, os.path.join(OUT_ROOT, STEP_DIRS[step_idx]), hash_id)

        # Save final (step 49)
        np_final = decode_final_video_to_np(final_video)
        save_img(np_final, os.path.join(OUT_ROOT, STEP_DIRS[49]), hash_id)

print("Done.")