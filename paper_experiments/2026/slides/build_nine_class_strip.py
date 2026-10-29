"""Nine-class strip for the slides: one real pool image per class (top row) and three LoRA text-to-image samples."""
import os, pandas as pd
from PIL import Image, ImageDraw, ImageFont
T = 150; GEN = "/dev/shm/qwen_lora_generations"; REALS = "/root/data/fitzpatrick17k/finalfitz17k"
nine = ["psoriasis", "lichen planus", "basal cell carcinoma", "allergic contact dermatitis", "lupus erythematosus",
        "neutrophilic dermatoses", "photodermatoses", "sarcoidosis", "squamous cell carcinoma"]
pool = pd.read_csv("/workspace/synderm/qwen_lora/generations/pool9.csv")
done = {h[:-4] for h in os.listdir(f"{GEN}/t2i/02") if h.endswith(".png")}
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
strip = Image.new("RGB", (T * 9, T * 4 + 40), "white"); dr = ImageDraw.Draw(strip)
for i, lab in enumerate(nine):
    hs = [h for h in pool[pool.label == lab].md5hash if h in done]
    assert hs, f"no t2i images for {lab}"
    h = hs[0]
    words = lab.split(" "); line1 = " ".join(words[:2]) if len(words) > 2 else words[0]; line2 = " ".join(words[2:]) if len(words) > 2 else (words[1] if len(words) > 1 else "")
    dr.text((i * T + 4, 4), line1, fill="black", font=font); dr.text((i * T + 4, 20), line2, fill="black", font=font)
    strip.paste(Image.open(f"{REALS}/{h}.jpg").convert("RGB").resize((T, T)), (i * T, 40))
    for k in range(3):
        strip.paste(Image.open(f"{GEN}/t2i/{k:02d}/{h}.png").convert("RGB").resize((T, T)), (i * T, 40 + (k + 1) * T))
strip.save("fig/nine_classes_t2i.jpg", quality=90); print("strip written")
