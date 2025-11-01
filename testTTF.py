from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# Create an image with noise background
# image_size = (512, 512)
# noise = (np.random.rand(*image_size, 3) * 255).astype('uint8')
# image = Image.fromarray(noise)

# # Apply a slight Gaussian blur to make the noise slightly blurred
# blur_radius = 1.0  # reduce for less blur, increase for more (e.g., 0.5..2.0)
# image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

# # Save the image
# image.save('output_image.png')

# Load the font
font_path = 'fonts/arial.ttf'  # Replace with your font file path
font_size = 100
font = ImageFont.truetype(font_path, font_size)

# Create an image with white background
image_size = (200, 200)
image = Image.new('RGB', image_size, 'white')
draw = ImageDraw.Draw(image)

# Define the character to display
character = 'W'  # Replace with the character you want to display

# Measure text size at origin, compute centered position
bbox0 = draw.textbbox((0, 0), character, font=font)
text_w = bbox0[2] - bbox0[0]
text_h = bbox0[3] - bbox0[1]
text_x = (image_size[0] - text_w) / 2
text_y = (image_size[1] - text_h) / 2

# Draw the character
draw.text((text_x, text_y), character, font=font, fill='black')

# Get the actual bounding box for the drawn text and draw a rectangle around it
actual_bbox = draw.textbbox((int(text_x), int(text_y)), character, font=font)
pad = 4  # padding around bbox in pixels
rect = (
    int(actual_bbox[0]) - pad,
    int(actual_bbox[1]) - pad,
    int(actual_bbox[2]) + pad,
    int(actual_bbox[3]) + pad,
)
draw.rectangle(rect, outline='red', width=2)

# Save the image
image.save('output_image.png')

