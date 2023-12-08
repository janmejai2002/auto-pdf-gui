import sys
import os
import shutil
# import fitz  # PyMuPDF
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QButtonGroup,QRadioButton, QProgressBar
from utils import process_pdf, process_pdf_box
from tqdm import tqdm
from PyQt6.QtCore import QThread, pyqtSignal

class WorkerThread(QThread):
    task_completed_signal = pyqtSignal(int, str)
    
    def __init__(self,selected_files, selected_option,color_setting, widget_instance):
        super().__init__()

        self.selected_files = selected_files
        self.selected_option = selected_option
        self.color_setting = color_setting
        self.widget_instance = widget_instance

    def run(self):
        if len(self.selected_files) == 0:
            pass
        
        if self.selected_option == 'note':
            for i, file in tqdm(enumerate(self.selected_files)):
                # print(f"Processing {file}...")

                bool_process = process_pdf(file, self.color_setting)
                self.task_completed_signal.emit(i+1, "")

                if not bool_process:
                    print(f"PDF not generated for {file}")
                
            
        if self.selected_option == 'box':
            for i, file in tqdm(enumerate(self.selected_files)):
                # print(f"Processing {file}...")
                bool_process = process_pdf_box(file)
                self.task_completed_signal.emit(i+1, "")


                if not bool_process:
                    print("PDF not generated")


        self.task_completed_signal.emit(len(self.selected_files), "completed")




class MyGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Auto-Notes')
        self.resize(600, 400)
        # Initialize variables  
        self.selected_files = []
        self.selected_option = "note"
        self.color_setting = "custom"

        self.color_values = self.color_selection()  # Default color values
        self.color_setting = "custom"
        
        
        # Set up the layout
        layout = QVBoxLayout()

        # Radio buttons for Note and Box
        option_label = QLabel("Select Option:")
        layout.addWidget(option_label)

        self.note_radio = QRadioButton("Note")
        self.box_radio = QRadioButton("Box")
        self.note_radio.setChecked(True)  # Set default selection
        layout.addWidget(self.note_radio)
        layout.addWidget(self.box_radio)

        # Button group for Note and Box
        self.option_button_group = QButtonGroup()
        self.option_button_group.addButton(self.note_radio)
        self.option_button_group.addButton(self.box_radio)

        # Connect radio button signals for Note and Box
        self.note_radio.toggled.connect(lambda: self.option_changed("note"))
        self.box_radio.toggled.connect(lambda: self.option_changed("box"))

        # Color setting labels and radio buttons
        color_setting_label = QLabel("Color Setting for Note: ")
        layout.addWidget(color_setting_label)

        self.color_yellow_radio = QRadioButton("Yellow")
        self.color_green_radio = QRadioButton("Green")
        self.color_custom_radio = QRadioButton("Custom")
        self.color_custom_radio.setChecked(True)
        layout.addWidget(self.color_yellow_radio)
        layout.addWidget(self.color_green_radio)
        layout.addWidget(self.color_custom_radio)

        # Button group for Yellow, Green, and Custom
        self.color_button_group = QButtonGroup()
        self.color_button_group.addButton(self.color_yellow_radio)
        self.color_button_group.addButton(self.color_green_radio)
        self.color_button_group.addButton(self.color_custom_radio)

        # Connect radio button signals for Yellow, Green, and Custom
        self.color_yellow_radio.toggled.connect(lambda: self.color_changed("yellow"))
        self.color_green_radio.toggled.connect(lambda: self.color_changed("green"))
        self.color_custom_radio.toggled.connect(lambda: self.color_changed("custom"))        # Color Display
        color_label = QLabel("Color:")
        layout.addWidget(color_label)
        self.color_display = QLabel()
        self.update_color_display()
        layout.addWidget(self.color_display)

        
        # File Selector
        self.file_label = QLabel("Selected File/Folder: ")
        layout.addWidget(self.file_label)
        self.file_label.setWordWrap(True)

        file_button = QPushButton("Select File/Folder")
        file_button.clicked.connect(self.show_file_dialog)
        layout.addWidget(file_button)

        # Run Button
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_main_code)
        layout.addWidget(self.run_button)
        
        
        #Progress Bar
        self.progress_label = QLabel("Progress : ")
        layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)
        
        self.completion_label = QLabel("Completion Status : ")
        layout.addWidget(self.completion_label)
        

        # PDF Viewer
        # self.pdf_viewer = QLabel()
        # layout.addWidget(self.pdf_viewer)

        # Set the layout for the main window
        self.setLayout(layout)
        # self.setWindowTitle("Auto-Notes")

    def show_file_dialog(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("PDF Files (*.pdf);;All Files (*)")
        if file_dialog.exec():
            self.selected_files = file_dialog.selectedFiles()
            base_names = [os.path.basename(file) for file in self.selected_files]
            self.file_label.setText(f"Selected Files: {', '.join(base_names)}")

            # Check if the selected file is a folder (optional)
            if os.path.isdir(self.selected_files[0]):
                pdf_files = [f for f in os.listdir(self.selected_files[0]) if f.lower().endswith('.pdf')]
                if pdf_files:
                    pdf_path = os.path.join(self.selected_files[0], pdf_files[0])
                    # self.display_pdf(pdf_path)
                else:
                    pass
                    # self.pdf_viewer.clear()
            else:
                pass
                # self.display_pdf(self.selected_files[0])

    def option_changed(self, option):
        self.selected_option = option
        if self.selected_option == 'box':
            self.color_green_radio.setCheckable(False)
            self.color_yellow_radio.setCheckable(False)
        if self.selected_option == 'note':
            self.color_green_radio.setCheckable(True)
            self.color_yellow_radio.setCheckable(True)
            
        self.color_values = self.color_selection()  # Update color values based on the selected option
        self.update_color_display()

    def color_changed(self, option):
        self.color_setting = option
        self.color_values = self.color_selection()  # Update color values based on the selected option
        self.update_color_display()


    def color_selection(self):
        if self.color_setting == "yellow" and self.selected_option == "note":
            return (255, 247, 106)
        elif self.color_setting == "green" and self.selected_option=='note':
            return (126, 253, 125)
        
        elif self.color_setting == "custom":
            filename = 'values_notes.log' if self.selected_option == 'note' else 'values.log'
            with open(filename) as file:
                lines = file.readlines()
                color = tuple(map(int, lines[0].strip()[1:-1].split(', ')))
                color = tuple(element + 40 for element in color)
                return color[::-1]


    def update_color_display(self):
        color_string = f"RGB: {self.color_values[0]}, {self.color_values[1]}, {self.color_values[2]}"
        self.color_display.setText(color_string)
        self.color_display.setStyleSheet(f"background-color: rgb({self.color_values[0]}, {self.color_values[1]}, {self.color_values[2]})")

    def cleanup(self):
        if os.path.exists('output'):
            shutil.rmtree('output')
        if os.path.exists('output_images'):
            shutil.rmtree('output_images')
    def run_main_code(self):
        self.run_button.setEnabled(False)
        self.progress_label.setText("Progress : ")
        self.completion_label.setText("Completion Status : ")
        self.progress_bar.setRange(0, len(self.selected_files))

        self.worker_thread = WorkerThread(selected_files=self.selected_files, selected_option=self.selected_option, color_setting=self.color_setting, widget_instance=self)
        self.worker_thread.task_completed_signal.connect(self.update_progress_and_completion)
        self.worker_thread.start()
        
        
    def update_progress_and_completion(self, value, completion_text):
        self.progress_label.setText(f"Progress : {value}/{len(self.selected_files)}")
        if completion_text != "completed":
            self.progress_bar.setValue(value)
        else:
            # If completion_text is 'completed', set the progress bar to 100%
            self.progress_bar.setValue(100) 

        if completion_text:
            self.completion_label.setText(f"Completion Status: {completion_text}")
            self.cleanup()
            # If completion_text is 'completed', update the random number label



        


    # def display_pdf(self, pdf_path):
    #     print(f"Displaying PDF: {pdf_path}")
    #     doc = fitz.open(pdf_path)
    #     page = doc.load_page(0)
    #     pix = page.get_pixmap()

    #     # Convert the pixmap to a PIL Image
    #     image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    #     image = image.resize((300, 400))  # Adjust the size as needed

    #     # Save the image for debugging purposes
    #     image.save('a.png')

    #     # Convert the PIL Image to a QImage
    #     qimage = self.pil_image_to_qimage(image)

    #     # Create a QPixmap from the QImage
    #     pixmap = QPixmap.fromImage(qimage)

    #     print(f"Image Size: width={pixmap.width()}, height={pixmap.height()}")
    #     self.pdf_viewer.setPixmap(pixmap)
        
    # def pil_image_to_qimage(self, pil_image):
    #     # Convert a PIL Image to a QImage
    #     pil_image = pil_image.convert("RGBA")  # Ensure it has an alpha channel
    #     qimage = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format.Format_RGBA8888)
    #     return qimage

if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_gui = MyGUI()
    my_gui.show()
    sys.exit(app.exec())
