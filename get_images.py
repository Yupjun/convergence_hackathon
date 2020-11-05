def get_images(image_folder):
    images = []
    for image_path in sorted(glob.glob(os.path.join(image_folder, image_format))):
        filename = os.path.basename(image_path)
        images.append((filename, image_path))
    return random.sample(images, 5)