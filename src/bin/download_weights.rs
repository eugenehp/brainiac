/// Download BrainIAC weights placeholder.
///
/// The actual BrainIAC weights must be converted from PyTorch .ckpt format
/// to .safetensors using `scripts/export_safetensors.py`.
///
/// This binary serves as a placeholder for future HuggingFace Hub integration.

fn main() {
    eprintln!("BrainIAC weight download");
    eprintln!();
    eprintln!("BrainIAC weights are distributed as PyTorch checkpoints.");
    eprintln!("To use with brainiac, convert them to safetensors format:");
    eprintln!();
    eprintln!("  1. Download from: https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/...");
    eprintln!("  2. Convert:  python scripts/export_safetensors.py \\");
    eprintln!("                 --input BrainIAC.ckpt \\");
    eprintln!("                 --output brainiac_backbone.safetensors");
    eprintln!();
    eprintln!("For downstream task heads, also convert the task checkpoint:");
    eprintln!("  python scripts/export_safetensors.py \\");
    eprintln!("    --input brainage_model.ckpt \\");
    eprintln!("    --output brainage_head.safetensors \\");
    eprintln!("    --head-only");
}
