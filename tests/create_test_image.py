from PIL import Image
import numpy as np

# Create a simple test image
width, height = 512, 512
# Create a gradient image for testing
x = np.linspace(0, 1, width)
y = np.linspace(0, 1, height)
X, Y = np.meshgrid(x, y)

# Create RGB channels with different patterns
r = (X * 255).astype(np.uint8)
g = (Y * 255).astype(np.uint8) 
b = ((X + Y) / 2 * 255).astype(np.uint8)

# Stack channels
rgb_array = np.stack([r, g, b], axis=-1)

# Create PIL image
test_image = Image.fromarray(rgb_array, 'RGB')
test_image.save('test_input.jpg', 'JPEG', quality=95)

print("✅ Created test_input.jpg")
