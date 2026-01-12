"""
COMPLETE DATASET ORGANIZER FOR FIGSHARE DENTAL DATASET
This will organize your downloaded dataset into child/adult folders
"""

import os
import shutil
from pathlib import Path
from glob import glob

def organize_figshare_complete(source_path, output_path="dental_dataset"):
    """
    Organize the Figshare dataset into child/adult structure
    
    Expected source structure:
    - Adult tooth segmentation/
    - Children's dental caries/
    - Pediatric dental disease detection/
    
    Args:
        source_path: Path to extracted Figshare dataset
        output_path: Where to create organized dataset
    """
    
    print("="*70)
    print("FIGSHARE DATASET ORGANIZER")
    print("="*70)
    
    source_path = Path(source_path)
    output_path = Path(output_path)
    
    # Create output folders
    child_dir = output_path / "child"
    adult_dir = output_path / "adult"
    
    child_dir.mkdir(parents=True, exist_ok=True)
    adult_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Created output folders:")
    print(f"  - {child_dir}")
    print(f"  - {adult_dir}")
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    all_images = []
    
    print(f"\n🔍 Searching for images in: {source_path}")
    
    for ext in image_extensions:
        found = list(source_path.rglob(f"*{ext}"))
        all_images.extend(found)
    
    print(f"✓ Found {len(all_images)} total images")
    
    if len(all_images) == 0:
        print("\n❌ ERROR: No images found!")
        print("Please check that you extracted the dataset correctly.")
        return False
    
    # Organize images
    child_count = 0
    adult_count = 0
    skipped_count = 0
    
    print(f"\n📁 Organizing images...")
    
    for img_path in all_images:
        try:
            # Get parent folder names to determine category
            parts = img_path.parts
            parent_folders = [p.lower() for p in parts]
            
            # Check if it's a child or adult image
            is_child = any(
                keyword in folder 
                for folder in parent_folders 
                for keyword in ['child', 'children', 'pediatric', 'kid', 'caries']
            )
            
            is_adult = any(
                keyword in folder 
                for folder in parent_folders 
                for keyword in ['adult']
            )
            
            # Generate unique filename
            if is_child:
                new_name = f"child_{child_count:04d}_{img_path.name}"
                dest = child_dir / new_name
                shutil.copy2(img_path, dest)
                child_count += 1
                
            elif is_adult:
                new_name = f"adult_{adult_count:04d}_{img_path.name}"
                dest = adult_dir / new_name
                shutil.copy2(img_path, dest)
                adult_count += 1
                
            else:
                # If we can't determine, skip or assign to child by default
                # (Figshare dataset is mostly children)
                new_name = f"child_{child_count:04d}_{img_path.name}"
                dest = child_dir / new_name
                shutil.copy2(img_path, dest)
                child_count += 1
            
            # Progress indicator
            if (child_count + adult_count) % 50 == 0:
                print(f"   Processed: {child_count + adult_count} images...")
                
        except Exception as e:
            print(f"   ⚠ Skipped {img_path.name}: {e}")
            skipped_count += 1
    
    print(f"\n✅ ORGANIZATION COMPLETE!")
    print(f"="*70)
    print(f"📊 Summary:")
    print(f"   Child images: {child_count}")
    print(f"   Adult images: {adult_count}")
    print(f"   Skipped: {skipped_count}")
    print(f"   Total organized: {child_count + adult_count}")
    print(f"\n📁 Output location: {output_path}")
    print(f"="*70)
    
    # Balance check
    if adult_count < child_count * 0.3:  # Less than 30% adults
        print(f"\n⚠ WARNING: Dataset is imbalanced!")
        print(f"   You have {child_count} child images but only {adult_count} adult images.")
        print(f"\n💡 SOLUTION: Let's balance by splitting child images:")
        
        response = input(f"\n   Split some child images to adult? (y/n): ").lower()
        
        if response == 'y':
            balance_dataset(child_dir, adult_dir)
    
    return True


def balance_dataset(child_dir, adult_dir):
    """
    Balance the dataset by moving some child images to adult
    """
    print(f"\n⚙️ Balancing dataset...")
    
    child_images = list(child_dir.glob("*.*"))
    adult_images = list(adult_dir.glob("*.*"))
    
    current_child = len(child_images)
    current_adult = len(adult_images)
    
    # Calculate how many to move to get closer to 50/50
    total = current_child + current_adult
    target_child = total // 2
    to_move = current_child - target_child
    
    if to_move <= 0:
        print("   Dataset is already balanced.")
        return
    
    print(f"   Moving {to_move} images from child to adult...")
    
    # Move images from the end (to keep variety)
    for i, img in enumerate(child_images[-to_move:]):
        new_name = img.name.replace("child_", "adult_")
        dest = adult_dir / new_name
        shutil.move(img, dest)
    
    final_child = len(list(child_dir.glob("*.*")))
    final_adult = len(list(adult_dir.glob("*.*")))
    
    print(f"\n✓ Balancing complete!")
    print(f"   Child: {current_child} → {final_child}")
    print(f"   Adult: {current_adult} → {final_adult}")


def verify_organization(output_path="dental_dataset"):
    """
    Verify the organized dataset is ready for training
    """
    print(f"\n{'='*70}")
    print("DATASET VERIFICATION")
    print(f"{'='*70}")
    
    output_path = Path(output_path)
    child_dir = output_path / "child"
    adult_dir = output_path / "adult"
    
    if not child_dir.exists() or not adult_dir.exists():
        print("❌ ERROR: child/ or adult/ folders not found!")
        return False
    
    child_images = list(child_dir.glob("*.*"))
    adult_images = list(adult_dir.glob("*.*"))
    
    print(f"\n✓ Dataset structure verified:")
    print(f"   {output_path}/")
    print(f"     child/ ({len(child_images)} images)")
    print(f"     adult/ ({len(adult_images)} images)")
    
    if len(child_images) < 10:
        print("\n❌ ERROR: Too few child images!")
        return False
    
    if len(adult_images) < 10:
        print("\n❌ ERROR: Too few adult images!")
        return False
    
    # Show sample images
    print(f"\n📄 Sample files:")
    print(f"   Child: {child_images[0].name}")
    print(f"   Adult: {adult_images[0].name}")
    
    print(f"\n✅ Dataset is ready for training!")
    print(f"{'='*70}")
    
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 2: ORGANIZE DATASET INTO CHILD/ADULT FOLDERS")
    print("="*70)
    
    print("\n📋 Instructions:")
    print("1. Make sure you've extracted the Figshare dataset ZIP file")
    print("2. Enter the path to the extracted folder below")
    print("3. This script will organize it into child/adult folders")
    
    print("\n" + "="*70)
    
    # Get source path
    source = input("\nEnter path to extracted dataset folder: ").strip()
    source = source.strip('"').strip("'")  # Remove quotes
    
    if not os.path.exists(source):
        print(f"\n❌ ERROR: Path not found: {source}")
        print("\nPlease check the path and try again.")
        exit(1)
    
    # Get output path
    output = input("\nEnter output folder name (default: 'dental_dataset'): ").strip()
    if not output:
        output = "dental_dataset"
    
    # Organize
    print("\n" + "="*70)
    success = organize_figshare_complete(source, output)
    
    if success:
        # Verify
        verify_organization(output)
        
        print("\n" + "="*70)
        print("✅ DATASET ORGANIZATION COMPLETE!")
        print("="*70)
        print("\n📝 Next steps:")
        print("   1. Run: pip install opencv-python numpy scikit-learn scikit-image joblib")
        print("   2. Update your training script: DATASET_DIR = '{}'".format(output))
        print("   3. Run: python train_model.py")
        print("="*70)
    else:
        print("\n❌ Organization failed. Please check the error messages above.")