from pathlib import Path
from PIL import Image

def get_image_dimensions(filter=False, max_width=32, max_height=32, base_path='data/raw'):
    """
    Get dimensions of all images in the specified directory structure.
    
    Args:
        base_path: Path to the directory containing class folders
    """
    base = Path(base_path)
    extensions = ('.png', '.jpeg', '.jpg', '.tif')
    
    results = []
    
    # Iterate through class folders
    for class_folder in base.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            
            # Iterate through images in each class folder
            for img_path in class_folder.iterdir():
                if img_path.suffix.lower() in extensions:
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            if filter:
                              if width < max_width or height < max_height:
                                results.append({
                                    'class': class_name,
                                    'filename': img_path.name,
                                    'width': width,
                                    'height': height,
                                    'path': str(img_path)
                                })
                                print(f"{class_name}/{img_path.name}: {width}x{height}")
                            else:
                              results.append({
                                  'class': class_name,
                                  'filename': img_path.name,
                                  'width': width,
                                  'height': height,
                                  'path': str(img_path)
                              })
                              print(f"{class_name}/{img_path.name}: {width}x{height}")
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
    
    return results

def calculate_statistics(image_data):
    """Calculate summary statistics for image dimensions."""
    if not image_data:
        print("No images found!")
        return
    
    widths = [img['width'] for img in image_data]
    heights = [img['height'] for img in image_data]
    
    stats = {
        'average_width': sum(widths) / len(widths),
        'average_height': sum(heights) / len(heights),
        'max_width': max(widths),
        'max_height': max(heights),
        'min_width': min(widths),
        'min_height': min(heights)
    }
    
    # Find images with max/min dimensions
    max_w_img = next(img for img in image_data if img['width'] == stats['max_width'])
    max_h_img = next(img for img in image_data if img['height'] == stats['max_height'])
    min_w_img = next(img for img in image_data if img['width'] == stats['min_width'])
    min_h_img = next(img for img in image_data if img['height'] == stats['min_height'])
    
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    print(f"Average dimensions: {stats['average_width']:.1f} x {stats['average_height']:.1f}")
    print(f"Maximum width: {stats['max_width']} ({max_w_img['class']}/{max_w_img['filename']})")
    print(f"Maximum height: {stats['max_height']} ({max_h_img['class']}/{max_h_img['filename']})")
    print(f"Minimum width: {stats['min_width']} ({min_w_img['class']}/{min_w_img['filename']})")
    print(f"Minimum height: {stats['min_height']} ({min_h_img['class']}/{min_h_img['filename']})")
    
    return stats

def calculate_class_statistics(image_data):
    """Calculate statistics per class."""
    classes = {}
    for img in image_data:
        class_name = img['class']
        if class_name not in classes:
            classes[class_name] = []
        classes[class_name].append(img)
    
    print("\n" + "="*60)
    print("STATISTICS BY CLASS")
    print("="*60)
    
    for class_name, images in classes.items():
        widths = [img['width'] for img in images]
        heights = [img['height'] for img in images]
        
        print(f"\n{class_name}:")
        print(f"  Count: {len(images)}")
        print(f"  Average dimensions: {sum(widths)/len(widths):.1f} x {sum(heights)/len(heights):.1f}")
        print(f"  Max dimensions: {max(widths)} x {max(heights)}")
        print(f"  Min dimensions: {min(widths)} x {min(heights)}")

if __name__ == '__main__':
    print("Getting image dimensions...\n")
    # image_data = get_image_dimensions()
    image_data = get_image_dimensions(filter=True, max_height=511, max_width=511)
    
    print(f"\nTotal images processed: {len(image_data)}")
    
    if image_data:
        calculate_statistics(image_data)
        calculate_class_statistics(image_data)