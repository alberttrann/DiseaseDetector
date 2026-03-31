import os
from pathlib import Path
from collections import defaultdict

def explore_directory_structure(base_path, max_depth=5):
    """
    Recursively explore directory structure and find all images
    """
    base_path = Path(base_path)
    
    print("=" * 100)
    print("EXPLORING DIRECTORY STRUCTURE")
    print("=" * 100)
    print(f"\nBase Directory: {base_path}")
    print(f"Absolute Path: {base_path.absolute()}\n")
    
    # Image extensions to look for
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', 
                       '.bmp', '.BMP', '.gif', '.GIF', '.tiff', '.TIFF'}
    
    # Statistics
    stats = {
        'total_dirs': 0,
        'total_files': 0,
        'total_images': 0,
        'image_by_extension': defaultdict(int),
        'images_by_dir': defaultdict(int),
        'all_image_paths': []
    }
    
    def get_tree_chars(is_last, depth):
        """Get tree drawing characters"""
        if depth == 0:
            return ""
        prefix = "    " * (depth - 1)
        return prefix + ("└── " if is_last else "├── ")
    
    def explore_recursive(path, depth=0, prefix=""):
        """Recursively explore directories"""
        if depth > max_depth:
            return
        
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            print(f"{prefix}[Permission Denied]")
            return
        
        dirs = [item for item in items if item.is_dir()]
        files = [item for item in items if item.is_file()]
        
        stats['total_dirs'] += len(dirs)
        stats['total_files'] += len(files)
        
        # Count images in this directory
        images_here = [f for f in files if f.suffix in image_extensions]
        if images_here:
            stats['images_by_dir'][str(path)] = len(images_here)
            stats['total_images'] += len(images_here)
            for img in images_here:
                stats['image_by_extension'][img.suffix] += 1
                stats['all_image_paths'].append(str(img))
        
        # Print directories
        for idx, directory in enumerate(dirs):
            is_last_dir = (idx == len(dirs) - 1) and len(files) == 0
            tree_char = get_tree_chars(is_last_dir, depth + 1)
            
            # Count images in subdirectories
            subdir_images = sum(1 for _ in directory.rglob('*') 
                              if _.is_file() and _.suffix in image_extensions)
            
            dir_info = f"{directory.name}/"
            if subdir_images > 0:
                dir_info += f" [{subdir_images} images]"
            
            print(f"{tree_char}{dir_info}")
            
            # Recurse into subdirectory
            if depth < max_depth:
                new_prefix = prefix + ("    " if is_last_dir else "│   ")
                explore_recursive(directory, depth + 1, new_prefix)
        
        # Print files (only if there are images)
        if images_here and depth < max_depth:
            # Show first few images as examples
            sample_size = min(3, len(images_here))
            for idx, file in enumerate(images_here[:sample_size]):
                is_last = (idx == len(images_here) - 1) and (idx < sample_size)
                tree_char = get_tree_chars(is_last, depth + 1)
                size_kb = file.stat().st_size / 1024
                print(f"{tree_char}{file.name} ({size_kb:.1f} KB)")
            
            if len(images_here) > sample_size:
                remaining = len(images_here) - sample_size
                tree_char = get_tree_chars(True, depth + 1)
                print(f"{tree_char}... and {remaining} more images")
    
    # Start exploration
    print(base_path.name + "/")
    explore_recursive(base_path, 0)
    
    # Print statistics
    print("\n" + "=" * 100)
    print("DIRECTORY STATISTICS")
    print("=" * 100)
    print(f"\nTotal Directories: {stats['total_dirs']}")
    print(f"Total Files: {stats['total_files']}")
    print(f"Total Images: {stats['total_images']}")
    
    if stats['total_images'] > 0:
        print("\n" + "-" * 100)
        print("Images by Extension:")
        print("-" * 100)
        for ext, count in sorted(stats['image_by_extension'].items(), 
                                key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_images']) * 100
            print(f"  {ext:10s}: {count:6d} images ({percentage:5.1f}%)")
        
        print("\n" + "-" * 100)
        print("Top 20 Directories with Most Images:")
        print("-" * 100)
        sorted_dirs = sorted(stats['images_by_dir'].items(), 
                           key=lambda x: x[1], reverse=True)[:20]
        for dir_path, count in sorted_dirs:
            # Shorten path for display
            rel_path = Path(dir_path).relative_to(base_path)
            print(f"  {count:6d} images: {rel_path}")
        
        # Save detailed report
        output_file = Path('directory_structure_report.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("COMPLETE DIRECTORY STRUCTURE REPORT\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"Base Directory: {base_path}\n")
            f.write(f"Total Images Found: {stats['total_images']}\n\n")
            
            f.write("-" * 100 + "\n")
            f.write("ALL DIRECTORIES WITH IMAGES:\n")
            f.write("-" * 100 + "\n")
            for dir_path, count in sorted(stats['images_by_dir'].items(), 
                                         key=lambda x: x[1], reverse=True):
                rel_path = Path(dir_path).relative_to(base_path)
                f.write(f"{count:6d} images: {rel_path}\n")
            
            f.write("\n" + "-" * 100 + "\n")
            f.write("IMAGE EXTENSIONS FOUND:\n")
            f.write("-" * 100 + "\n")
            for ext, count in sorted(stats['image_by_extension'].items()):
                f.write(f"{ext}: {count} images\n")
            
            # Save first 100 image paths as examples
            f.write("\n" + "-" * 100 + "\n")
            f.write("SAMPLE IMAGE PATHS (first 100):\n")
            f.write("-" * 100 + "\n")
            for img_path in stats['all_image_paths'][:100]:
                rel_path = Path(img_path).relative_to(base_path)
                f.write(f"{rel_path}\n")
            
            if len(stats['all_image_paths']) > 100:
                f.write(f"\n... and {len(stats['all_image_paths']) - 100} more images\n")
        
        print(f"\n✓ Detailed report saved to: {output_file.absolute()}")
        
        # Save CSV with all image paths
        csv_file = Path('all_image_paths.csv')
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("relative_path,absolute_path,filename,extension,size_kb,parent_dir\n")
            for img_path in stats['all_image_paths']:
                img_path = Path(img_path)
                rel_path = img_path.relative_to(base_path)
                size_kb = img_path.stat().st_size / 1024
                f.write(f'"{rel_path}","{img_path}","{img_path.name}",'
                       f'"{img_path.suffix}",{size_kb:.2f},"{img_path.parent.name}"\n')
        
        print(f" CSV with all image paths saved to: {csv_file.absolute()}")
    
    else:
        print("\n  No images found in the directory structure!")
        print("\nChecked for extensions:", ', '.join(sorted(image_extensions)))
    
    print("\n" + "=" * 100)
    
    return stats


def find_potential_dataset_dirs(base_path):
    """Find directories that might contain the actual dataset"""
    base_path = Path(base_path)
    
    print("\n" + "=" * 100)
    print("SEARCHING FOR POTENTIAL DATASET LOCATIONS")
    print("=" * 100)
    
    keywords = ['rice', 'disease', 'crop', 'plant', 'leaf', 'pest', 
                'healthy', 'train', 'test', 'images', 'data']
    
    potential_dirs = []
    
    for item in base_path.rglob('*'):
        if item.is_dir():
            name_lower = item.name.lower()
            if any(keyword in name_lower for keyword in keywords):
                # Count images in this directory
                image_count = sum(1 for _ in item.glob('**/*') 
                                if _.is_file() and _.suffix.lower() in 
                                {'.jpg', '.jpeg', '.png', '.bmp'})
                if image_count > 0:
                    potential_dirs.append((item, image_count))
    
    if potential_dirs:
        print("\nFound directories with relevant names and images:")
        for dir_path, count in sorted(potential_dirs, key=lambda x: x[1], reverse=True):
            rel_path = dir_path.relative_to(base_path)
            print(f"  {count:6d} images: {rel_path}")
    else:
        print("\nNo directories with dataset-related names found.")
    
    return potential_dirs


if __name__ == "__main__":
    # Get the current directory
    base_dir = Path.cwd()
    
    print("DIRECTORY STRUCTURE EXPLORER")
    
    print(f"Current Working Directory: {base_dir}")
    print(f"Directory exists: {base_dir.exists()}")
    print(f"Is directory: {base_dir.is_dir()}")
    
    if not base_dir.exists():
        print("\n Directory does not exist!")
    else:
        # Explore structure
        stats = explore_directory_structure(base_dir, max_depth=4)
        
        # Find potential dataset locations
        if stats['total_images'] == 0:
            find_potential_dataset_dirs(base_dir)
        
        print("EXPLORATION COMPLETE!")
