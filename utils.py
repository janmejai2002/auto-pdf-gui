import os, cv2, shutil, fitz, img2pdf
import numpy as np
from PIL import Image, ImageGrab
import os
import tkinter as tk
from tkinter import filedialog, ttk, Label
import threading
from pynput import keyboard, mouse 

def get_max_image_dimensions(folder_path):
    max_width = 0
    max_height = 0

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)

            if image is not None:
                height, width, _ = image.shape
                max_width = max(max_width, width)
                max_height = max(max_height, height)

    return max_width


def stitch_all_boxes(folder_path, image_list):
    images = []
    output_width = get_max_image_dimensions(folder_path=folder_path)
    top_padding = 10
    bottom_padding = 10

    for filename in image_list:
        image = cv2.imread(filename)

        if image is not None:
            current_width = image.shape[1]
            border_width = output_width - current_width
            left_border_width = border_width // 2

            canvas_height = image.shape[0] + top_padding + bottom_padding
            canvas = np.ones((canvas_height, output_width, 3), np.uint8) * 255

            # Add the image to the canvas with padding
            canvas[top_padding:top_padding + image.shape[0], left_border_width:left_border_width + current_width, :] = image

            images.append(canvas)
    try:
        combined_image = np.vstack(images)
        return combined_image
    except Exception as e:
        return False

def stitch_all(folder_path, image_list):
    images = []
    output_width = get_max_image_dimensions(folder_path=folder_path)
    for filename in image_list:
        image = cv2.imread(filename)

        if image is not None:
            current_width = image.shape[1]
            border_width = output_width - current_width
            left_border_width = border_width // 2

            canvas = np.ones((image.shape[0], output_width, 3), np.uint8) * 255
            canvas[:, left_border_width:left_border_width + current_width, :] = image
            images.append(canvas)

    try:
        combined_image = np.vstack(images)
        return combined_image
    except Exception as e:
        return False


def get_document(image, name):
    # Load the input image using cv2
      # Directory to save split images
    output_1 = os.path.join('output', name)
    output_dir = os.path.join(output_1, 'document_images')
    split_height = 1555  # Height at which to split the image

    if image is None:
        print("Error: Image not found.")
        return

    height, width, channels = image.shape

    # Check if the split height is within the image's height bounds
    if split_height <= 0 or split_height >= height:
        print("Error: Invalid split height. Image generated")
        leftover = split_height-height
        blank = np.ones((leftover, width, 3), dtype=np.uint8)*255
        output = np.vstack([image, blank])
        cv2.imwrite('output.jpg', output)
        return 

    # Determine the number of splits
    num_splits = height // split_height

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Split the image and save each part
    imgs = []
    start = 0
    split_count = 0

    while start < height:
        end = min(start + split_height, height)
        split_image = image[start:end, :]
        # Save the split image
        output_path = os.path.join(output_dir, f"split_{split_count}.png")
        cv2.imwrite(output_path, split_image)

        # print(f"Saved: {output_path}")
        imgs.append(output_path)
        split_count += 1
        start = end

    if not os.path.exists('outputpdf'):
        os.mkdir('outputpdf')
    output_pdf = os.path.join('outputpdf', f'{name}_notes.pdf')   # Specify the output PDF path

    with open(output_pdf, 'wb') as pdf_output:
        pdf_output.write(img2pdf.convert(imgs))
    print(f"Output Generated -> {output_pdf}")


def resize_image(clean_image):
    screen_height, screen_width = 720, 1080  # Adjust the screen_height as needed
    window_height, window_width = clean_image.shape[:2]

    if window_height > screen_height:
        # Calculate the width to maintain the aspect ratio
        aspect_ratio = window_width / window_height
        new_height = screen_height
        new_width = int(new_height * aspect_ratio)
        clean_image = cv2.resize(clean_image, (new_width, new_height))
    return clean_image


def resize_img(image):
    r = 1080 / image.shape[1]
    dim = (1080 ,int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


# Define a function to remove consecutive rows of all-white pixel
def remove_consecutive_white_rows(image):
    result = []
    white_row_count = 0
    max_consecutive_white_rows = 10  # Adjust as needed

    for row in image:
        if np.all(row == 255):  # Check if the row is all-white
            white_row_count += 1
        else:
            white_row_count = 0

        if white_row_count <= max_consecutive_white_rows:
            result.append(row)

    return np.array(result)


def get_bounds(values):
    b, g ,r = values 
    lower = [b-40, g-40, r-40]
    upper = [b+10, g+10, r+10]
    l_clip=[max(0,min(value, 255)) for value in lower]
    u_clip=[max(0,min(value, 255)) for value in upper]
    lower_arr = np.array(l_clip, dtype=np.uint8)
    upper_arr = np.array(u_clip, dtype=np.uint8)
    return (lower_arr, upper_arr)

def select_pdf():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    return file_path

def process_image_box(page, counter, output_dir, output_masks):
    # Read the input image
    image_list = page.get_pixmap(matrix = fitz.Matrix(2, 2))
    image = Image.frombytes("RGB", [image_list.width, image_list.height], image_list.samples)
    image_np = np.array(image)
    image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    with open('values.log') as file:
        lines = file.readlines()

    # Remove leading and trailing whitespaces, then split and convert to integers
    lower_color = tuple(map(int, lines[0].strip()[1:-1].split(', ')))
    upper_color = tuple(map(int, lines[1].strip()[1:-1].split(', ')))

    # Convert to NumPy arrays
    lower_color = np.array(lower_color, dtype=np.uint8)
    upper_color = np.array(upper_color, dtype=np.uint8)

    #Values for test PDF
    # lower_color = np.array([40,115, 70], dtype=np.uint8)
    # upper_color = np.array([110, 185, 150], dtype=np.uint8)
    # print(lower_color, upper_color)

    # Threshold the image to isolate the specified color
    color_mask = cv2.inRange(image, lower_color, upper_color)

    cv2.imwrite(os.path.join(output_masks, f'colormask_{counter+1}.jpg'), color_mask)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    # Loop through identified contours and crop rectangles
    for idx, contour in (enumerate(contours)):
        epsilon = 0.02 * cv2.arcLength(contour, True)
        # epsilon = 0.03 * cv2.arcLength(contour, True) # testing
        
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4 and cv2.contourArea(contour):
            x, y, w, h = cv2.boundingRect(contour)
            if w>400:
                cropped_rectangle = image[y:y + h, x:x + w]

                # Save the cropped rectangle to the output directory
                cv2.imwrite(os.path.join(output_dir, f'page_{counter+1}_{idx}.jpg'), cropped_rectangle)

            else:
                pass

def clean_loop(image):
    while True:
        new_image = remove_consecutive_white_rows(image)
        if np.array_equal(new_image, image):
            break  # No more consecutive white rows to remove
        image = new_image

    return image

def process_image(page, color_setting):
    
    image_list = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    image = Image.frombytes("RGB", [image_list.width, image_list.height], image_list.samples)
    image_np = np.array(image)

    # Convert BGR to RGB format (if needed)
    image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    if color_setting == 'yellow' :
        lower_color=(66, 207, 215)
        upper_color=(116, 255, 255)

    if color_setting == 'green':
        lower_color=(85, 213, 86)
        upper_color=(135, 255, 136)        
        
    else:
        with open('values_notes.log') as file:
            lines = file.readlines()
        
        lower_color = tuple(map(int, lines[0].strip()[1:-1].split(', ')))
        upper_color = tuple(map(int, lines[1].strip()[1:-1].split(', ')))
        
    lower_color = np.array(lower_color, dtype=np.uint8)
    upper_color = np.array(upper_color, dtype=np.uint8)

    color_mask = cv2.inRange(image, lower_color, upper_color)
    #method old
    # kernel = np.ones((3, 3), np.uint8)
    # openning = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # kernel = np.ones((8, 8), np.uint8)
    # closing = cv2.dilate(openning, kernel, iterations=3)

    # preserved_highlights = cv2.bitwise_and(image, image, mask=closing)
    # gray = cv2.cvtColor(preserved_highlights, cv2.COLOR_RGB2GRAY)
    # _, preserved_text = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY_INV)
    # preserved_text = cv2.bitwise_and(preserved_text, preserved_text, mask=closing)

    # final_image = 255 - preserved_text
    
    #method new
    kernel = np.ones((3, 3), np.uint8)
    imd = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=10)
    dilate = cv2.dilate(imd, kernel=kernel, iterations=2)
    preserved = cv2.bitwise_and(image, image, mask = dilate)

    gray = cv2.cvtColor(preserved, cv2.COLOR_RGB2GRAY)
    _, preserved_text = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY_INV)
    preserved_text = cv2.bitwise_and(preserved_text, preserved_text, mask=dilate)
    final_image = 255 - preserved_text
    cleaned_image = clean_loop(final_image)

    resized_img = resize_img(cleaned_image)
    return resized_img


def get_pdf_info(input_pdf):
    pdf_document = fitz.open(input_pdf)
    total = pdf_document.page_count
    return total


def get_image_list(output_dir):
    images = os.listdir(output_dir)
    images.sort(key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
    image_paths = [os.path.join(output_dir, image) for image in images]
    if not image_paths:
        print("No valid images were processed.")
        return None, None
    return image_paths

def get_max_image_dimensions(folder_path):
    max_width = 0
    max_height = 0

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)

            if image is not None:
                height, width, _ = image.shape
                max_width = max(max_width, width)
                max_height = max(max_height, height)

    return max_width

def process_pdf(input_pdf, color_setting):

    name = os.path.splitext(os.path.basename(input_pdf))[0]

    output_dir_name = os.path.join('output', name)  # Specify the output directory
    output_dir = os.path.join(output_dir_name, 'process_images')
    os.makedirs(output_dir, exist_ok=True)

    pdf_document = fitz.open(input_pdf)
    total_pages = pdf_document.page_count
    for page_number in range(total_pages):
        page = pdf_document.load_page(page_number)
        processed_image = process_image(page, color_setting) 
        if processed_image is not None and processed_image.shape[0] > 3:
            image_path = os.path.join(output_dir, f'page_{page_number + 1}.jpg')
            cv2.imwrite(image_path, processed_image)
        else:
            pass
    pdf_document.close()
    
    image_paths = get_image_list(output_dir)
    if not image_paths:
        print("No valid images generated. Please check values")
        exit()
    
    combined_image = stitch_all(output_dir, image_paths)

    if combined_image is None:
        return False
    else:
        get_document(combined_image, name)
        return True

def process_pdf_box(input_pdf):
    name = os.path.splitext(os.path.basename(input_pdf))[0]
    output_dir = os.path.join('output', name)
    output_images = os.path.join(output_dir, 'final_images')
    output_masks = os.path.join(output_dir, 'masks')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_masks, exist_ok=True)

    pdf_document = fitz.open(input_pdf)
    total_pages = pdf_document.page_count

    for page_number in range(total_pages):
        page = pdf_document.load_page(page_number)
        process_image_box(page, page_number, output_images, output_masks)

    pdf_document.close()

    image_paths = get_image_list(output_images)
    if not image_paths:
        print("No valid images generated. Please check your values.")
        exit()

    combined_image = stitch_all_boxes(output_images, image_paths)
    if combined_image is None:
        return False
    else:
        get_document(combined_image, name)
        return True
    

def show_instruction(img):

    border_color = [255, 255, 255]  # White color
    border_size = 40  # Adjust as needed

    # Create a white strip
    white_strip = np.ones((border_size, img.shape[1], 3), dtype=np.uint8) * border_color

    # Stack the white strip on top of the image
    img_with_strip = np.vstack((white_strip, img))

    # Add text to the image
    text_lines = [
        "Find Image with box",
        "   Press Enter to Select",
        "   Press L to Next Image"
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2
    font_color = (0, 0, 0)  # Black color
    text_position = (10, 10)  # Adjust the starting position as needed

    for text_line in text_lines:
        cv2.putText(
            img_with_strip, text_line,
            text_position, font, font_scale,
            font_color, font_thickness, cv2.LINE_AA
        )
        text_position = (text_position[0], text_position[1] + 30)  # Adjust the vertical spacing

    return img_with_strip


class ColorPickerApp:
    value_logger = []  # Class variable to store color values

    def __init__(self):
        self.color_picker_enabled = False

        # Create the main window
        self.root = tk.Tk()
        self.root.title("Color Picker")

        # Create GUI elements
        self.status_label = Label(self.root, text="Status : ")
        self.status_label.grid(row=0, column=0, padx=2, pady=10)

        self.color_picker_button = ttk.Button(self.root, text="Off", command=self.toggle_color_picker)
        self.color_picker_button.grid(row=1, column=0, padx=10, pady=10)

        self.info_label = ttk.Label(self.root, text="Click above to turn on")
        self.info_label.grid(row=2, column=0, padx=10, pady=10)

        self.rgb_label = ttk.Label(self.root, text="")
        self.rgb_label.grid(row=3, column=0, padx=10, pady=10)

        self.color_display = Label(self.root, text="", bg='#F0F0F0', width=5, height=5)
        self.color_display.grid(row=4, column=0, padx=10, pady=10)

        self.exit_button = ttk.Button(self.root, text="Done", command=self.exit_app)
        self.exit_button.grid(row=5, column=0, padx=10, pady=10)

        # Set up keyboard and mouse listeners in a separate thread
        self.listener_thread = threading.Thread(target=self.start_listeners, daemon=True)
        self.listener_thread.start()

    def start_listeners(self):
        with keyboard.Listener(on_release=self.on_release) as klstnr:
            with mouse.Listener(on_click=self.on_click) as mlstnr:
                klstnr.join()
                mlstnr.join()

    def toggle_color_picker(self):
        self.color_picker_enabled = not self.color_picker_enabled
        if self.color_picker_enabled:
            self.info_label.config(text="Please click on the color pixel.")
            self.color_picker_button.config(text="On")
        else:
            self.info_label.config(text="Click above to turn on")
            self.color_picker_button.config(text="Off")

    def on_click(self, x, y, button, pressed):
        if self.color_picker_enabled and pressed and button == mouse.Button.left:
            try:
                self.check_color(x, y)
            except Exception as e:
                print(f"Error while checking color: {e}")

    def check_color(self, x, y):
        bbox = (x, y, x + 1, y + 1)
        im = ImageGrab.grab(bbox=bbox)
        rgbim = im.convert('RGB')
        r, g, b = rgbim.getpixel((0, 0))
        output = f'COLOR: rgb{(r, g, b)} | HEX #{self.get_hex((r, g, b))}\n'
        self.color_display.config(bg=f"#{self.get_hex((r, g, b))}")
        self.rgb_label.config(text=output)
        self.__class__.value_logger.append((b, g, r))

    def get_hex(self, rgb):
        return '%02X%02X%02X' % rgb

    def on_release(self, key):
        if key == keyboard.Key.esc:
            self.root.destroy()

    def exit_app(self):
        # ... (file handling code if uncommented)
        self.color_picker_enabled = False
        self.root.destroy()

    @classmethod
    def get_value_logger(cls):
        if len(cls.value_logger) == 0:
            print("Nothing selected")
        elif len(cls.value_logger) == 1:
            return cls.value_logger[0]
        else:
            return cls.value_logger[-2]

import cv2
import numpy as np
from PIL import Image
import fitz
from utils import *

def box_finder():
    def process_image_box(image, lower, upper):
        mask = cv2.inRange(image, lower, upper)
        return mask

    def on_slider_change(x):
        # Get current slider values
        lower = (cv2.getTrackbarPos('Lower B', 'Trackbars'),
                    cv2.getTrackbarPos('Lower G', 'Trackbars'),
                    cv2.getTrackbarPos('Lower R', 'Trackbars'))
        upper = (cv2.getTrackbarPos('Upper B', 'Trackbars'),
                    cv2.getTrackbarPos('Upper G', 'Trackbars'),
                    cv2.getTrackbarPos('Upper R', 'Trackbars'))

        # Process the image with the updated values
        mask = process_image_box(image, lower, upper)
        resized_mask = resize_image(mask)
        resized_image = resize_image(image)

        colored_mask = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((resized_image, colored_mask))

        # return (lower, upper, resized)
        # Show the processed image
        cv2.imshow('Only white rectangle should be visible', combined)

        # cv2.imshow('Only Rectangle Should be Visible', resized_mask)


    def create_trackbars():
        cv2.createTrackbar('Lower B', 'Trackbars', 0, 255, on_slider_change)
        cv2.createTrackbar('Lower G', 'Trackbars', 0, 255, on_slider_change)
        cv2.createTrackbar('Lower R', 'Trackbars', 0, 255, on_slider_change)

        cv2.createTrackbar('Upper B', 'Trackbars', 0, 255, on_slider_change)
        cv2.createTrackbar('Upper G', 'Trackbars', 0, 255, on_slider_change)
        cv2.createTrackbar('Upper R', 'Trackbars', 0, 255, on_slider_change)

        cv2.setTrackbarPos('Lower B', 'Trackbars', init_lower[0])
        cv2.setTrackbarPos('Lower G', 'Trackbars', init_lower[1])
        cv2.setTrackbarPos('Lower R', 'Trackbars', init_lower[2])

        cv2.setTrackbarPos('Upper B', 'Trackbars', init_upper[0])
        cv2.setTrackbarPos('Upper G', 'Trackbars', init_upper[1])
        cv2.setTrackbarPos('Upper R', 'Trackbars', init_upper[2])


    def get_final_values():
        # Get current slider values
        lower = (cv2.getTrackbarPos('Lower B', 'Trackbars'),
                    cv2.getTrackbarPos('Lower G', 'Trackbars'),
                    cv2.getTrackbarPos('Lower R', 'Trackbars'))
        upper = (cv2.getTrackbarPos('Upper B', 'Trackbars'),
                    cv2.getTrackbarPos('Upper G', 'Trackbars'),
                    cv2.getTrackbarPos('Upper R', 'Trackbars'))
        # print(f"Lower BGR - {lower}\nUpper BGR - {upper}")
        with open('values.log', 'w') as f:
            f.write(f"{lower}\n")
            f.write(f"{upper}")
        print("Values saved to values.log")


    app = ColorPickerApp()
    app.root.mainloop()
    result = app.get_value_logger()
    del app

    input_pdf = select_pdf()
    if not input_pdf:
        print('No PDF selected')
        exit()

    pdf_document = fitz.open(input_pdf)
    selected = False
    page_number=0
    while not selected:
        page = pdf_document.load_page(page_number)
        image_list = page.get_pixmap(matrix = fitz.Matrix(300/72, 300/72))
        image = Image.frombytes("RGB", [image_list.width, image_list.height], image_list.samples)
        image_np = np.array(image)
        image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_r = resize_image(image)
        image_x = show_instruction(image_r)
        image_x = image_x.astype(np.uint8)
        
        cv2.imshow("Select Image for Testing Range", image_x)

        key = cv2.waitKey(0) & 0xFF

        if key == 13:
            selected = True
            cv2.destroyAllWindows()
        elif key == 76 or key==108:
            page_number+=1

    if image is None:
        print("No image selected. Exiting.")
        exit()

    cv2.namedWindow('Trackbars')

    init_lower, init_upper = get_bounds(result) 

    create_trackbars()

    cv2.waitKey(0)
    get_final_values()
    cv2.destroyAllWindows()

def highlight_finder():
    def resize_image(clean_image):
        screen_height, screen_width = 720, 1080  # Adjust the screen_height as needed
        window_height, window_width = clean_image.shape[:2]

        if window_height > screen_height:
            # Calculate the width to maintain the aspect ratio
            aspect_ratio = window_width / window_height
            new_height = screen_height
            new_width = int(new_height * aspect_ratio)
            clean_image = cv2.resize(clean_image, (new_width, new_height))
        return clean_image

    def process_image(image, lower_green, upper_green, k1, k2, i1, i2, t1):
        green_mask = cv2.inRange(image, lower_green, upper_green)
        kernel = np.ones((k1, k1), np.uint8)
        openning = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=i1)
        kernel2 = np.ones((k2, k2), np.uint8)
        closing = cv2.dilate(openning, kernel2, iterations=i2)
        preserved_highlights = cv2.bitwise_and(image, image, mask=closing)
        gray = cv2.cvtColor(preserved_highlights, cv2.COLOR_RGB2GRAY)
        _, preserved_text = cv2.threshold(gray, t1, 255, cv2.THRESH_BINARY_INV)
        preserved_text = cv2.bitwise_and(preserved_text, preserved_text, mask=closing)
        final_image = 255 - preserved_text

        return final_image

    def on_slider_change(x):
        # Get current slider values
        lower_green = (cv2.getTrackbarPos('Lower B', 'Trackbars'),
                    cv2.getTrackbarPos('Lower G', 'Trackbars'),
                    cv2.getTrackbarPos('Lower R', 'Trackbars'))
        upper_green = (cv2.getTrackbarPos('Upper B', 'Trackbars'),
                    cv2.getTrackbarPos('Upper G', 'Trackbars'),
                    cv2.getTrackbarPos('Upper R', 'Trackbars'))

        k1 = (cv2.getTrackbarPos('k1', 'Trackbars'))
        k2 = (cv2.getTrackbarPos('k2', 'Trackbars'))
        i1 =  (cv2.getTrackbarPos('i1', 'Trackbars'))
        i2 = (cv2.getTrackbarPos('i2', 'Trackbars'))
        t1 =  (cv2.getTrackbarPos('t1', 'Trackbars'))
        
        resized_input_image = resize_image(image)
        mask = process_image(image, lower_green, upper_green, k1, k2, i1, i2, t1)
        resized_mask = resize_image(mask)

        colored_mask = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)

        combined = np.hstack((resized_input_image, colored_mask))

        cv2.imshow('Image', combined)

    def create_trackbars():

        init_k1 = 3
        init_k2 = 8
        init_i1 = 2
        init_i2 = 3
        init_t1 = 128
        # Create sliders for lower and upper green values
        cv2.createTrackbar('Lower B', 'Trackbars', 0, 255, on_slider_change)
        cv2.createTrackbar('Lower G', 'Trackbars', 0, 255, on_slider_change)
        cv2.createTrackbar('Lower R', 'Trackbars', 0, 255, on_slider_change)

        cv2.createTrackbar('Upper B', 'Trackbars', 0, 255, on_slider_change)
        cv2.createTrackbar('Upper G', 'Trackbars', 0, 255, on_slider_change)
        cv2.createTrackbar('Upper R', 'Trackbars', 0, 255, on_slider_change)

        cv2.createTrackbar('k1', 'Trackbars', 0, 10, on_slider_change)
        cv2.createTrackbar('k2', 'Trackbars', 0, 10, on_slider_change)
        cv2.createTrackbar('i1', 'Trackbars', 0, 10, on_slider_change)
        cv2.createTrackbar('i2', 'Trackbars', 0, 10, on_slider_change)
        cv2.createTrackbar('t1', 'Trackbars', 0, 255, on_slider_change)

        # Create sliders for lower and upper green values
        cv2.setTrackbarPos('Lower B', 'Trackbars', init_lower_green[0])
        cv2.setTrackbarPos('Lower G', 'Trackbars', init_lower_green[1])
        cv2.setTrackbarPos('Lower R', 'Trackbars', init_lower_green[2])

        cv2.setTrackbarPos('Upper B', 'Trackbars', init_upper_green[0])
        cv2.setTrackbarPos('Upper G', 'Trackbars', init_upper_green[1])
        cv2.setTrackbarPos('Upper R', 'Trackbars', init_upper_green[2])

        cv2.setTrackbarPos('k1', 'Trackbars', init_k1)
        cv2.setTrackbarPos('k2', 'Trackbars', init_k2)
        cv2.setTrackbarPos('i1', 'Trackbars', init_i1)
        cv2.setTrackbarPos('i2', 'Trackbars', init_i2)
        cv2.setTrackbarPos('t1', 'Trackbars', init_t1)

    def get_final_values():
        # Get current slider values
        lower = (cv2.getTrackbarPos('Lower B', 'Trackbars'),
                    cv2.getTrackbarPos('Lower G', 'Trackbars'),
                    cv2.getTrackbarPos('Lower R', 'Trackbars'))
        upper = (cv2.getTrackbarPos('Upper B', 'Trackbars'),
                    cv2.getTrackbarPos('Upper G', 'Trackbars'),
                    cv2.getTrackbarPos('Upper R', 'Trackbars'))
        # print(f"Lower BGR - {lower}\nUpper BGR - {upper}")
        with open('values_notes.log', 'w') as f:
            f.write(f"{lower}\n")
            f.write(f"{upper}")
        print("Values saved to values_notes.log")
        


    app= ColorPickerApp()
    app.root.mainloop()
    result=app.get_value_logger()
    del app

    input_pdf = select_pdf()
    if not input_pdf:
        print("No PDF Selected")
        exit()

    pdf_document  = fitz.open(input_pdf)
    selected  = False 
    page_number = 0
    while not selected:
        page = pdf_document.load_page(page_number)
        image_list = page.get_pixmap(matrix = fitz.Matrix(300/72, 300/72))
        image = Image.frombytes("RGB", [image_list.width, image_list.height], image_list.samples)
        image_np = np.array(image)
        image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_r = resize_image(image)
        image_x = show_instruction(image_r)
        image_x = image_x.astype(np.uint8)
        
        cv2.imshow("Select Image for Testing Range", image_x)

        key = cv2.waitKey(0) & 0xFF

        if key == 13:
            selected = True
            cv2.destroyAllWindows()
        elif key == 76 or key==108:
            page_number+=1

    if image is None:
        print("No image selected. Exiting.")
        exit()

    # Create a window for trackbars
    cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)

    init_lower_green, init_upper_green = get_bounds(result)

    create_trackbars()

    # Process the image with initial values
    # clean_image = process_image(image, init_lower_green, init_upper_green, init_k1, init_k2, init_i1, init_i2, init_t1)
    # resized_image = resize_image(clean_image=clean_image)
    # Display the processed image
    # cv2.imshow('Processed Image', resized_image)

    cv2.waitKey(0)
    get_final_values()
    cv2.destroyAllWindows()
