"""Arc2Face wrapper (frozen identity-conditioned diffusion model, SD1.5 backbone).
Written against the Arc2Face README; verify the calls if the repository changes."""
import os, sys, torch


def download_weights(arc_root):
    from huggingface_hub import hf_hub_download, snapshot_download
    os.makedirs(f"{arc_root}/models", exist_ok=True)
    for f in ["arc2face/config.json", "arc2face/diffusion_pytorch_model.safetensors",
              "encoder/config.json", "encoder/pytorch_model.bin"]:
        hf_hub_download("FoivosPaspalis/Arc2Face", f, local_dir=f"{arc_root}/models")
    # antelopev2 ArcFace/detector. Check the Arc2Face README if this repo id changes.
    snapshot_download("DIAMONDLX/antelopev2", local_dir=f"{arc_root}/models/antelopev2")


class Arc2FaceGenerator:
    def __init__(self, arc_root, device="cuda", base="stable-diffusion-v1-5/stable-diffusion-v1-5"):
        sys.path.append(arc_root)
        from arc2face import CLIPTextModelWrapper, project_face_embs
        from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
        self.device, self._project = device, project_face_embs
        enc = CLIPTextModelWrapper.from_pretrained(f"{arc_root}/models", subfolder="encoder", torch_dtype=torch.float16)
        unet = UNet2DConditionModel.from_pretrained(f"{arc_root}/models", subfolder="arc2face", torch_dtype=torch.float16)
        self.pipe = StableDiffusionPipeline.from_pretrained(base, text_encoder=enc, unet=unet,
                                                            torch_dtype=torch.float16, safety_checker=None).to(device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def generate(self, z, seeds, steps=25, guidance=3.0):
        """z: unit-norm 512-d identity vector (numpy). Returns one PIL image per seed."""
        emb = torch.tensor(z, dtype=torch.float16, device=self.device)[None]
        pe = self._project(self.pipe, emb)
        gens = [torch.Generator(self.device).manual_seed(int(s)) for s in seeds]
        return self.pipe(prompt_embeds=pe, num_inference_steps=steps, guidance_scale=guidance,
                         num_images_per_prompt=len(seeds), generator=gens).images
