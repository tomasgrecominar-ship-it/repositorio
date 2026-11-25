import sys, json, time, torch, os
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, AutoencoderKL
S=os.environ["S"]; M=f"{S}/models/realvis"
torch.set_num_threads(4)
vae=AutoencoderKL.from_pretrained(f"{M}/vae", torch_dtype=torch.float32)
pipe=StableDiffusionXLPipeline.from_pretrained(M, vae=vae, variant="fp16", torch_dtype=torch.bfloat16)
pipe.vae.to(torch.float32)
pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
NEG="cartoon, painting, illustration, 3d render, cgi, deformed, disfigured, bad anatomy, extra fingers, mutated hands, extra limbs, blurry face, lowres, text, watermark, logo, oversaturated, ugly"
jobs=json.load(open(sys.argv[1]))
for j in jobs:
    out=f"{S}/img/{j['name']}_s{j['seed']}.png"
    if os.path.exists(out): continue
    t=time.time()
    g=torch.Generator().manual_seed(j['seed'])
    with torch.inference_mode():
        lat=pipe(prompt=j['p'], negative_prompt=NEG, width=768, height=1344, num_inference_steps=6, guidance_scale=1.0, generator=g, output_type="latent").images
        x=pipe.vae.decode(lat.to(torch.float32)/pipe.vae.config.scaling_factor).sample
        im=pipe.image_processor.postprocess(x, output_type="pil")[0]
    im.save(out); print(j['name'], j['seed'], f"{time.time()-t:.0f}s", flush=True)
