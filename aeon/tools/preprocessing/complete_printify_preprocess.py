import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import io

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("WARNING: rembg not available. Background removal will be skipped.")

def process_image(input_path: str, output_path: str, args):
    """Process a single image through the pipeline."""
    print(f"Processing: {input_path}")
    
    try:
        # Open image
        img = Image.open(input_path)
        
        # Convert to RGBA for transparency support
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Stage 1: Resize if specified
        if args.target_width or args.target_height:
            orig_w, orig_h = img.size
            if args.target_width and args.target_height:
                new_w, new_h = args.target_width, args.target_height
            elif args.target_width:
                scale = args.target_width / orig_w
                new_w = args.target_width
                new_h = int(orig_h * scale)
            else:
                scale = args.target_height / orig_h
                new_w = int(orig_w * scale)
                new_h = args.target_height
            
            img = img.resize((new_w, new_h), Image.LANCZOS)
            print(f"  Resized to {new_w}x{new_h}")
        
        # Stage 2: Background removal
        if args.background_removal and REMBG_AVAILABLE:
            print("  Removing background...")
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Try multiple models if first fails
            for model in ['u2net', 'u2netp', 'silueta']:
                try:
                    img = remove(img_bytes.read(), model_name=model)
                    img = Image.open(io.BytesIO(img))
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    print(f"  Background removed using {model}")
                    break
                except Exception as e:
                    print(f"    Model {model} failed: {e}")
                    continue
            else:
                print("  WARNING: All background removal models failed")
        
        # Stage 3: Trim transparent edges
        if args.trim_transparent_edges:
            print("  Trimming transparent edges...")
            img = trim_transparent_edges(img)
        
        # Stage 4: Add watermark if specified
        if args.watermark_text:
            print(f"  Adding watermark: {args.watermark_text}")
            img = add_watermark(img, args.watermark_text)
        
        # Stage 5: Save with correct DPI and format
        output_format = args.output_format.upper()
        if output_format == 'JPG' or output_format == 'JPEG':
            # Convert to RGB for JPG
            if img.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Save with DPI
        save_kwargs = {'dpi': (args.dpi, args.dpi)}
        if output_format in ['PNG', 'TIFF']:
            save_kwargs['format'] = output_format
        elif output_format in ['JPG', 'JPEG']:
            save_kwargs['format'] = 'JPEG'
            save_kwargs['quality'] = 95
        
        img.save(output_path, **save_kwargs)
        print(f"  Saved to {output_path} ({img.size[0]}x{img.size[1]}, {args.dpi} DPI)")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return False


def trim_transparent_edges(img: Image.Image) -> Image.Image:
    """Trim transparent edges from an RGBA image."""
    if img.mode != 'RGBA':
        return img
    
    # Get alpha channel
    alpha = img.split()[3]
    
    # Find bounding box of non-transparent pixels
    bbox = alpha.getbbox()
    
    if bbox is None:
        # All transparent, return empty image
        return Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    
    # Crop to bounding box
    return img.crop(bbox)


def add_watermark(img: Image.Image, text: str) -> Image.Image:
    """Add a watermark text to the image."""
    from PIL import ImageDraw, ImageFont
    
    # Create a copy to draw on
    img = img.copy()
    
    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(img)
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position in bottom-right corner with padding
    padding = 10
    x = img.size[0] - text_width - padding
    y = img.size[1] - text_height - padding
    
    # Draw semi-transparent background for text
    background_bbox = (x - 5, y - 5, x + text_width + 5, y + text_height + 5)
    draw.rectangle(background_bbox, fill=(0, 0, 0, 128))
    
    # Draw text
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    
    return img


def main():
    parser = argparse.ArgumentParser(description='Printify Image Preprocessing Pipeline')
    parser.add_argument('--input_dir', required=True, help='Input directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--background_removal', action='store_true', default=True, help='Remove background')
    parser.add_argument('--no_background_removal', action='store_true', help='Disable background removal')
    parser.add_argument('--target_width', type=int, help='Target width in pixels')
    parser.add_argument('--target_height', type=int, help='Target height in pixels')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI')
    parser.add_argument('--trim_transparent_edges', action='store_true', default=True, help='Trim transparent edges')
    parser.add_argument('--no_trim', action='store_true', help='Disable edge trimming')
    parser.add_argument('--output_format', default='PNG', help='Output format (PNG/JPG)')
    parser.add_argument('--watermark_text', type=str, help='Watermark text to add')
    parser.add_argument('--max_workers', type=int, default=2, help='Max parallel workers')
    
    args = parser.parse_args()
    
    # Override flags
    if args.no_background_removal:
        args.background_removal = False
    if args.no_trim:
        args.trim_transparent_edges = False
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    images = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]
    
    if not images:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(images)} images to process")
    
    # Process each image
    success_count = 0
    for img_path in images:
        output_path = output_dir / f"{img_path.stem}_processed.{args.output_format.lower()}"
        if process_image(str(img_path), str(output_path), args):
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(images)} images processed successfully")


if __name__ == '__main__':
    main()