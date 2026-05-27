import sys
import shutil
from pathlib import Path
import numpy as np
import blindspot as bs
import click
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                               QComboBox, QCheckBox, QGraphicsView, QGraphicsScene, 
                               QGraphicsPixmapItem, QFrame, QSplitter, QFileDialog, 
                               QGraphicsRectItem, QGraphicsItem, QStyleOptionGraphicsItem)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QWheelEvent, QAction
from PySide6.QtCore import Qt, Signal, QObject, QRectF, QLineF

def to_qimage(image: np.ndarray) -> QImage:
    """Convert numpy array to QImage"""
    image = (image * 255.0).astype(np.uint8)
    if len(image.shape) == 2:
        height, width = image.shape
        bytes_per_line = width
        return QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    else:
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        return QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

class GlobalOverlayItem(QGraphicsItem):
    def __init__(self, bad_map):
        super().__init__()
        self.bad_map = bad_map
        self.rows, self.cols = bad_map.shape
        self.bad_rects = []
        
        # Pre-calculate bad rects
        ys, xs = np.where(bad_map == 255)
        for y, x in zip(ys, xs):
            self.bad_rects.append(QRectF(x, y, 1, 1))
            
    def boundingRect(self):
        return QRectF(0, 0, self.cols, self.rows)
        
    def paint(self, painter, option, widget):
        # Optimization: Only draw grid in exposed rect
        rect = option.exposedRect
        left = int(max(0, rect.left()))
        top = int(max(0, rect.top()))
        right = int(min(self.cols, rect.right() + 1))
        bottom = int(min(self.rows, rect.bottom() + 1))
        
        # Draw Green Grid
        green_pen = QPen(QColor('green'))
        green_pen.setWidth(0) # Cosmetic
        painter.setPen(green_pen)
        
        lines = []
        # Vertical lines
        for x in range(left, right + 1):
             lines.append(QLineF(x, top, x, bottom))
        # Horizontal lines
        for y in range(top, bottom + 1):
             lines.append(QLineF(left, y, right, y))
             
        if lines:
            painter.drawLines(lines)
        
        # Draw Red Bad Pixels
        # Optimization: Only draw bad rects that intersect visible area
        # But if bad_rects list is small, iterating it is cheap.
        # If it's large, we might want a spatial index, but for now linear scan is likely fine
        # unless there are 10k+ bad pixels.
        red_pen = QPen(QColor('red'))
        red_pen.setWidth(0)
        painter.setPen(red_pen)
        
        visible_bad_rects = []
        for r in self.bad_rects:
            if r.intersects(rect):
                visible_bad_rects.append(r)
        
        if visible_bad_rects:
            painter.drawRects(visible_bad_rects)

class ImageGraphicsView(QGraphicsView):
    """Custom GraphicsView for image display with zoom and pan support"""
    pixelSelected = Signal(int, int) # y, x

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = None
        self.selection_items = []
        self._zoom = 0
        self.setBackgroundBrush(QColor(50, 50, 50))
        self.setFrameShape(QFrame.NoFrame)

    def set_image(self, qimage):
        self.scene.clear()
        self.selection_items = []
        pixmap = QPixmap.fromImage(qimage)
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def draw_selection(self, y, x, bad_map, range_size=21):
        # Clear previous selection
        for item in self.selection_items:
            self.scene.removeItem(item)
        self.selection_items = []

        # Draw Blue Dashed Cursor for the selected pixel only
        pen = QPen(QColor('blue'))
        pen.setStyle(Qt.DashLine)
        pen.setWidth(2) 
        pen.setCosmetic(True) 
        
        cursor_rect = QGraphicsRectItem(x, y, 1, 1)
        cursor_rect.setPen(pen)
        cursor_rect.setZValue(100) # Ensure it's on top of everything
        self.scene.addItem(cursor_rect)
        self.selection_items.append(cursor_rect)

    def wheelEvent(self, event: QWheelEvent):
        """Handle pan with mouse wheel (touchpad scroll)"""
        # Mac touchpad scroll usually provides both angleDelta().x() and y()
        # We use these deltas to scroll the viewport
        h_delta = event.angleDelta().x()
        v_delta = event.angleDelta().y()
        
        # Adjust sensitivity as needed
        scroll_speed = 1.0 
        
        # Scroll the horizontal scrollbar
        h_bar = self.horizontalScrollBar()
        h_bar.setValue(h_bar.value() - int(h_delta * scroll_speed))
        
        # Scroll the vertical scrollbar
        v_bar = self.verticalScrollBar()
        v_bar.setValue(v_bar.value() - int(v_delta * scroll_speed))
        
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            # Right click to select pixel
            pos = self.mapToScene(event.position().toPoint())
            x, y = int(pos.x()), int(pos.y())
            if self.pixmap_item and 0 <= x < self.pixmap_item.pixmap().width() and 0 <= y < self.pixmap_item.pixmap().height():
                self.pixelSelected.emit(y, x)
        elif event.button() == Qt.LeftButton:
            # Store position for click vs drag detection
            self._drag_start_pos = event.position().toPoint()
            
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and hasattr(self, '_drag_start_pos'):
            # Check if it was a click (not a drag)
            delta = (event.position().toPoint() - self._drag_start_pos).manhattanLength()
            if delta < 3: # Threshold for click
                pos = self.mapToScene(event.position().toPoint())
                x, y = int(pos.x()), int(pos.y())
                if self.pixmap_item and 0 <= x < self.pixmap_item.pixmap().width() and 0 <= y < self.pixmap_item.pixmap().height():
                    self.pixelSelected.emit(y, x)
        super().mouseReleaseEvent(event)

class MainWindow(QMainWindow):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.setWindowTitle("Blind Pixels Manual Search (PySide6)")
        self.resize(kwargs['window_width'], kwargs['window_height'])
        
        # Data initialization
        self.base_src = Path(kwargs['base_src'])
        bs.BASE_PATH = self.base_src
        self.current_proj_info = None
        self.overlay_item = None
        self.show_overlay = False
        self.selected_y = 0
        self.selected_x = 0

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5) # Reduce main layout margins
        main_layout.setSpacing(5)

        # Top Configuration Panel
        self.create_top_panel(main_layout)

        # Content Splitter (Left: Image, Right: Result)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1) # stretch=1 to take remaining space

        # Left Panel - Image View
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.image_view = ImageGraphicsView()
        self.image_view.pixelSelected.connect(self.on_pixel_selected)
        left_layout.addWidget(self.image_view)
        
        # Tools under image
        tools_layout = QHBoxLayout()
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.setToolTip("Shortcut: i")
        self.zoom_in_btn.clicked.connect(lambda: self.image_view.scale(1.25, 1.25))
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.setToolTip("Shortcut: o")
        self.zoom_out_btn.clicked.connect(lambda: self.image_view.scale(0.8, 0.8))
        tools_layout.addWidget(self.zoom_in_btn)
        tools_layout.addWidget(self.zoom_out_btn)
        left_layout.addLayout(tools_layout)
        
        splitter.addWidget(left_widget)

        # Right Panel - Result View
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.info_label = QLabel("Select a pixel to view details")
        right_layout.addWidget(self.info_label)
        
        self.result_view = QGraphicsView() # Placeholder for matplotlib or result image
        right_layout.addWidget(self.result_view)
        splitter.addWidget(right_widget)
        
        splitter.setSizes([600, 600])

        # Initialize Project List
        self.refresh_proj_list()

        # Set focus to image view by default
        self.image_view.setFocus()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_I:
            self.image_view.scale(1.25, 1.25)
        elif event.key() == Qt.Key_O:
            self.image_view.scale(0.8, 0.8)
        elif event.key() == Qt.Key_P:
            self.toggle_overlay()
        elif event.key() == Qt.Key_M:
            self.image_mode_cb.toggle()
        elif self.current_proj_info is not None:
            # Navigation keys
            step = 1
            new_y, new_x = self.selected_y, self.selected_x
            
            if event.key() == Qt.Key_Left:
                new_x -= step
            elif event.key() == Qt.Key_Right:
                new_x += step
            elif event.key() == Qt.Key_Up:
                new_y -= step
            elif event.key() == Qt.Key_Down:
                new_y += step
            else:
                super().keyPressEvent(event)
                return
                
            # Clamp
            height, width = self.noice.shape
            new_x = max(0, min(new_x, width - 1))
            new_y = max(0, min(new_y, height - 1))
            
            if new_x != self.selected_x or new_y != self.selected_y:
                self.on_pixel_selected(new_y, new_x)
                # Ensure the new selection is visible
                # We need to map coordinates to scene pos
                # Since pixels are 1x1 at (0,0), scene pos is just x, y
                self.image_view.ensureVisible(new_x, new_y, 1, 1, 50, 50)
        else:
            super().keyPressEvent(event)
            
    def toggle_overlay(self):
        if not self.overlay_item:
            return
        
        self.show_overlay = not self.show_overlay
        if self.show_overlay:
            if self.overlay_item not in self.image_view.scene.items():
                self.image_view.scene.addItem(self.overlay_item)
            self.overlay_item.setVisible(True)
        else:
            self.overlay_item.setVisible(False)

    def create_top_panel(self, parent_layout):
        top_panel = QFrame()
        # Compact layout: Horizontal layout containing both Base Src and Project
        top_layout = QHBoxLayout(top_panel)
        top_layout.setContentsMargins(0, 0, 0, 0) # Zero margins
        
        # Base Path Config
        top_layout.addWidget(QLabel("Base Src:"))
        self.path_entry = QLineEdit(str(self.base_src))
        top_layout.addWidget(self.path_entry, stretch=1)
        path_btn = QPushButton("Choose")
        path_btn.clicked.connect(self.choose_directory)
        top_layout.addWidget(path_btn)

        # Spacing
        top_layout.addSpacing(10)

        # Project Selection
        top_layout.addWidget(QLabel("Project:"))
        self.proj_combo = QComboBox()
        self.proj_combo.currentIndexChanged.connect(self.load_project)
        self.proj_combo.setMinimumWidth(200) # Ensure it's not too small
        top_layout.addWidget(self.proj_combo, stretch=1) 

        # Active Checkbox
        self.active_checkbox = QCheckBox("Active")
        self.active_checkbox.toggled.connect(self.on_active_toggled)
        top_layout.addWidget(self.active_checkbox)

        # Window Size Selection
        top_layout.addWidget(QLabel("Window Size:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(["3x3", "5x5", "7x7"])
        
        # Set initial index based on kwargs['r']
        initial_r = self.kwargs.get('r', 2)
        if initial_r == 1:
            self.size_combo.setCurrentIndex(0)
        else:
            self.size_combo.setCurrentIndex(1) # Default to 5x5 (r=2)
            # Force r=2 if it was something else (e.g. 3) to match the UI
            if initial_r != 2:
                self.kwargs['r'] = 2
            
        self.size_combo.currentIndexChanged.connect(self.on_window_size_changed)
        top_layout.addWidget(self.size_combo)

        # Show Axes Checkbox
        self.show_axis_cb = QCheckBox("Show Axes")
        self.show_axis_cb.setChecked(True) # Default ON
        self.show_axis_cb.toggled.connect(self.on_show_axis_toggled)
        top_layout.addWidget(self.show_axis_cb)

        # Absolute Y-Axis Checkbox
        self.absolute_y_cb = QCheckBox("Absolute Y")
        self.absolute_y_cb.setChecked(False) # Default OFF (Relative)
        self.absolute_y_cb.toggled.connect(self.on_absolute_y_toggled)
        top_layout.addWidget(self.absolute_y_cb)

        # Image Mode Checkbox (Variance/Mean)
        self.image_mode_cb = QCheckBox("Show Mean") # Default unchecked = Show Variance
        self.image_mode_cb.toggled.connect(self.on_image_mode_toggled)
        top_layout.addWidget(self.image_mode_cb)

        # Save Button
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_current_plot)
        top_layout.addWidget(self.save_btn)

        parent_layout.addWidget(top_panel, 0) # stretch=0 to prevent vertical expansion

    def on_show_axis_toggled(self, checked):
        if self.current_proj_info:
             self.on_pixel_selected(self.selected_y, self.selected_x)

    def on_absolute_y_toggled(self, checked):
        if self.current_proj_info:
             self.on_pixel_selected(self.selected_y, self.selected_x)

    def on_image_mode_toggled(self, checked):
        if not self.current_proj_info:
            return
            
        # Preserve current view state (zoom and scroll)
        current_transform = self.image_view.transform()
        h_bar = self.image_view.horizontalScrollBar()
        v_bar = self.image_view.verticalScrollBar()
        h_val = h_bar.value()
        v_val = v_bar.value()
            
        # checked = True -> Show Mean (Voltage)
        # checked = False -> Show Variance (Noise)
        if checked:
            qimg = to_qimage(bs.norm(self.average_image))
            self.image_mode_cb.setText("Show Variance") # Label indicates next action or current state? Let's keep it "Show Mean"
        else:
            qimg = to_qimage(bs.norm(self.noice))
            self.image_mode_cb.setText("Show Mean")
            
        self.image_view.set_image(qimg)
        
        # Restore view state
        self.image_view.setTransform(current_transform)
        h_bar.setValue(h_val)
        v_bar.setValue(v_val)
        
        # Re-add overlay if needed
        self.overlay_item = GlobalOverlayItem(self.bad)
        if self.show_overlay:
            self.image_view.scene.addItem(self.overlay_item)
            self.overlay_item.setVisible(True)
        else:
            self.overlay_item.setVisible(False)
            
        # Re-draw selection
        self.image_view.draw_selection(self.selected_y, self.selected_x, self.bad)

    def on_window_size_changed(self, index):
        if index == 0:
            self.kwargs['r'] = 1
        elif index == 1:
            self.kwargs['r'] = 2
        elif index == 2:
            self.kwargs['r'] = 3
        
        if self.current_proj_info:
            self.on_pixel_selected(self.selected_y, self.selected_x)

    def save_current_plot(self):
        if not self.current_proj_info:
            return
        
        # Source file (last generated plot)
        src_path = Path(self.kwargs['save_dir']) / f"{self.current_proj_info['index']}" / f"{self.selected_y}_{self.selected_x}.png"
        if not src_path.exists():
            return

        # Default save dir
        default_dir = Path("/Users/charles/workspace/library/blindspot/assets")
        if not default_dir.exists():
            default_dir.mkdir(parents=True, exist_ok=True)
            
        # Default filename
        default_filename = f"{self.current_proj_info['index']}_{self.selected_y}_{self.selected_x}.png"
        
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Plot", str(default_dir / default_filename), "Images (*.png)")
        
        if save_path:
            shutil.copy2(src_path, save_path)

    def refresh_proj_list(self):
        self.proj_combo.clear()
        if not self.base_src.exists():
            return
        
        proj_infos = bs.get_all_proj_info()
        for _, info in proj_infos.items():
            active_mark = "[ON] " if info.get('active', True) else "[OFF]"
            self.proj_combo.addItem(f"{active_mark} {info['index']}:{info['path']}", info)
        
        if self.proj_combo.count() > 4:
            self.proj_combo.setCurrentIndex(4) # Default as per original code

    def on_active_toggled(self, checked):
        if self.current_proj_info:
            self.current_proj_info['active'] = checked
            bs.change_info(self.current_proj_info)
            print(f"Project {self.current_proj_info['index']} active set to {checked}")
            
            # Update list item text dynamically
            idx = self.proj_combo.currentIndex()
            if idx >= 0:
                active_mark = "[ON] " if checked else "[OFF]"
                new_text = f"{active_mark} {self.current_proj_info['index']}:{self.current_proj_info['path']}"
                self.proj_combo.setItemText(idx, new_text)
                self.proj_combo.setItemData(idx, self.current_proj_info)

    def load_project(self, index):
        if index < 0: return
        data = self.proj_combo.currentData()
        
        # Reload latest info from disk to get fresh 'active' status
        fresh_data = bs.get_proj_info_by_index(data['index'])
        if fresh_data:
            data = fresh_data
            # Update the combobox data to keep it in sync
            self.proj_combo.setItemData(index, data)
            # Update item text to match fresh data (in case changed externally)
            active_mark = "[ON] " if data.get('active', True) else "[OFF]"
            new_text = f"{active_mark} {data['index']}:{data['path']}"
            self.proj_combo.setItemText(index, new_text)
            
        self.current_proj_info = data
        
        # Update Active Checkbox without triggering signal
        self.active_checkbox.blockSignals(True)
        self.active_checkbox.setChecked(data.get('active', True))
        self.active_checkbox.blockSignals(False)
        
        # Load data using blindspot library
        bs.load_low_noice(data)
        bs.load_high_noice(data)
        bs.load_high_voltages(data)
        bs.load_low_voltages(data)
        
        self.noice = data['noice_l']
        self.noice_35 = data['noice_h']
        self.voltage = data['vol_l']
        self.voltage_35 = data['vol_h']
        self.bad = bs.read_png_to_array(Path(data['path']) / 'BadPixel.png')
        
        self.average_noice = np.average(self.noice)
        self.average_noice_35 = np.average(self.noice_35)
        self.average_image = np.average(self.voltage, axis=0)
        self.average_image_35 = np.average(self.voltage_35, axis=0)

        # Display image (This clears the scene!)
        qimg = to_qimage(bs.norm(self.noice))
        self.image_view.set_image(qimg)
        self.info_label.setText(f"Project Loaded: {data['index']}")
        
        # Initialize and add overlay AFTER setting image
        # This prevents the overlay item from being deleted by set_image's scene.clear()
        self.overlay_item = GlobalOverlayItem(self.bad)
        if self.show_overlay:
            self.image_view.scene.addItem(self.overlay_item)
            self.overlay_item.setVisible(True)
        else:
            self.overlay_item.setVisible(False)

    def choose_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Choose Directory", str(self.base_src))
        if dir_path:
            self.base_src = Path(dir_path)
            self.path_entry.setText(str(self.base_src))
            bs.BASE_PATH = self.base_src
            self.refresh_proj_list()

    def on_pixel_selected(self, y, x):
        self.selected_y = y
        self.selected_x = x
        val = self.noice[y, x]
        is_bad = self.bad[y, x] == 255
        self.info_label.setText(f"Pixel: h={y}, w={x} | Value: {val:.4f} | Is Bad: {is_bad}")
        
        # Draw selection rectangle
        self.image_view.draw_selection(y, x, self.bad)

        # Draw wave
        bs.plot_wave(
            p = (y, x),
            r = self.kwargs['r'],
            bad = self.bad,
            proj_id = self.current_proj_info['index'],
            save_dir = Path(self.kwargs['save_dir']),
            double_temp = self.kwargs['double_temp'],
            vol_l= self.voltage,
            vol_h= self.voltage_35,
            vol_l_avg = self.average_image,
            vol_h_avg = self.average_image_35,
            noice_l_avg = self.average_noice,
            noice_h_avg = self.average_noice_35,
            show_axis = self.show_axis_cb.isChecked(),
            absolute_y = self.absolute_y_cb.isChecked()
        )
        
        # Load and display result image
        res_img_path = Path(self.kwargs['save_dir']) / f"{self.current_proj_info['index']}" / f"{y}_{x}.png"
        if res_img_path.exists():
            pixmap = QPixmap(str(res_img_path))
            scene = QGraphicsScene()
            scene.addPixmap(pixmap)
            self.result_view.setScene(scene)
            self.result_view.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)

@click.command()
@click.option('--base_src', default='/Volumes/Charles/data/blindpoint/source')
@click.option('--save_dir', default='/Volumes/Charles/data/blindpoint/tmp')
@click.option('--r', type=int, default=2)
@click.option('--double_temp', type=bool, default=True)
@click.option('--window_width', type=int, default=1200)
@click.option('--window_height', type=int, default=820)
def main(**kwargs):
    bs.set_base_path(kwargs['base_src'])
    app = QApplication(sys.argv)
    window = MainWindow(**kwargs)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
