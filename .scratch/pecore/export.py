"""Export PE-Core-S16-384 (open_clip) vision + text towers to ONNX. Throwaway spike."""
import sys, time, torch, open_clip

HUB = sys.argv[1] if len(sys.argv) > 1 else "hf-hub:timm/PE-Core-S-16-384"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/Users/magic/Creations/ImgTag/models"
TAG = sys.argv[3] if len(sys.argv) > 3 else "pecore-s16-384"
RES = int(sys.argv[4]) if len(sys.argv) > 4 else 384
CTX = int(sys.argv[5]) if len(sys.argv) > 5 else 32

t0 = time.time()
model, _, preprocess = open_clip.create_model_and_transforms(HUB)
model.eval()
print(f"loaded in {time.time()-t0:.1f}s; preprocess={preprocess}")
nv = sum(p.numel() for p in model.visual.parameters())
print(f"vision params: {nv/1e6:.2f}M | total: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")


class Vis(torch.nn.Module):
    def __init__(s, m):
        super().__init__(); s.m = m
    def forward(s, pixel_values):
        return s.m.encode_image(pixel_values)


class Txt(torch.nn.Module):
    def __init__(s, m):
        super().__init__(); s.m = m
    def forward(s, input_ids):
        return s.m.encode_text(input_ids)


with torch.no_grad():
    px = torch.randn(1, 3, RES, RES)
    v_ref = Vis(model)(px)
    print("vision out:", tuple(v_ref.shape), v_ref.dtype)
    torch.onnx.export(
        Vis(model), (px,), f"{OUT}/{TAG}-vision.onnx",
        input_names=["pixel_values"], output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print("VISION EXPORT OK")

    ids = torch.randint(0, 49407, (1, CTX), dtype=torch.int64)
    t_ref = Txt(model)(ids)
    print("text out:", tuple(t_ref.shape), t_ref.dtype)
    torch.onnx.export(
        Txt(model), (ids,), f"{OUT}/{TAG}-text.onnx",
        input_names=["input_ids"], output_names=["text_embeds"],
        dynamic_axes={"input_ids": {0: "batch"}, "text_embeds": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print("TEXT EXPORT OK")
print(f"total {time.time()-t0:.1f}s")
