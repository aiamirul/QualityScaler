import ctypes
import functools
import itertools
import multiprocessing
import os
import os.path
import platform
import shutil
import sys
import threading
import time
import webbrowser
from math import ceil, floor, sqrt
from multiprocessing.pool import ThreadPool
from timeit import default_timer as timer

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch_directml
from moviepy.editor import VideoFileClip
from moviepy.video.io import ImageSequenceClip
from PIL import Image

import os
#show only gpu index 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sv_ttk
global app_name
app_name = "QualityScaler"

models_array          = [ 'BSRGANx4', 'BSRGANx2', 'RealSR_JPEGx4' ]
AI_model              = models_array[0]

image_path            = "none"
device                = 1
input_video_path      = ""
target_file_extension = ".png"
file_extension_list   = [ '.png', '.jpg', '.jp2', '.bmp', '.tiff' ]
half_precision        = True
single_image          = False
multiple_images       = False
video_file            = False
multi_img_list        = []
video_frames_list     = []
frames_upscaled_list  = []
vram_multiplier       = 1
default_vram_limiter  = 8
multiplier_num_tiles  = 4
cpu_number            = 4
interpolation_mode    = cv2.INTER_LINEAR
windows_subversion    = int(platform.version().split('.')[2])
compatible_gpus       = torch_directml.device_count()

device_list_names     = []
device_list           = []

class Gpu:
    def __init__(self, name, index):
        self.name = name
        self.index = index


for index in range(compatible_gpus): 
    gpu = Gpu(name = torch_directml.device_name(index), index = index)
    device_list.append(gpu)
    device_list_names.append(gpu.name)

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)



supported_file_list     = ['.jpg', '.jpeg', '.JPG', '.JPEG',
                            '.png', '.PNG',
                            '.webp', '.WEBP',
                            '.bmp', '.BMP',
                            '.tif', '.tiff', '.TIF', '.TIFF',
                            '.mp4', '.MP4',
                            '.webm', '.WEBM',
                            '.mkv', '.MKV',
                            '.flv', '.FLV',
                            '.gif', '.GIF',
                            '.m4v', ',M4V',
                            '.avi', '.AVI',
                            '.mov', '.MOV',
                            '.qt', '.3gp', '.mpg', '.mpeg']

supported_video_list    = ['.mp4', '.MP4',
                            '.webm', '.WEBM',
                            '.mkv', '.MKV',
                            '.flv', '.FLV',
                            '.gif', '.GIF',
                            '.m4v', ',M4V',
                            '.avi', '.AVI',
                            '.mov', '.MOV',
                            '.qt',
                            '.3gp', '.mpg', '.mpeg']

not_supported_file_list = ['.txt', '.exe', '.xls', '.xlsx', '.pdf',
                           '.odt', '.html', '.htm', '.doc', '.docx',
                           '.ods', '.ppt', '.pptx', '.aiff', '.aif',
                           '.au', '.bat', '.java', '.class',
                           '.csv', '.cvs', '.dbf', '.dif', '.eps',
                           '.fm3', '.psd', '.psp', '.qxd',
                           '.ra', '.rtf', '.sit', '.tar', '.zip',
                           '.7zip', '.wav', '.mp3', '.rar', '.aac',
                           '.adt', '.adts', '.bin', '.dll', '.dot',
                           '.eml', '.iso', '.jar', '.py',
                           '.m4a', '.msi', '.ini', '.pps', '.potx',
                           '.ppam', '.ppsx', '.pptm', '.pst', '.pub',
                           '.sys', '.tmp', '.xlt', '.avif']


ctypes.windll.shcore.SetProcessDpiAwareness(True)
scaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
font_scale = round(1/scaleFactor, 1)


# ------------------- Slice functions -------------------


class Tile(object):
    def __init__(self, image, number, position, coords, filename=None):
        self.image = image
        self.number = number
        self.position = position
        self.coords = coords
        self.filename = filename

    @property
    def row(self): return self.position[0]

    @property
    def column(self): return self.position[1]

    @property
    def basename(self): return get_basename(self.filename)

    def generate_filename(
        self, directory=os.getcwd(), prefix="tile", format="png", path=True
    ):
        filename = prefix + "_{col:02d}_{row:02d}.{ext}".format(
            col=self.column, row=self.row, ext=format.lower().replace("jpeg", "jpg")
        )
        if not path: return filename
        return os.path.join(directory, filename)

    def save(self, filename=None, format="png"):
        if not filename: filename = self.generate_filename(format=format)
        self.image.save(filename, format)
        self.filename = filename

    def __repr__(self):
        """Show tile number, and if saved to disk, filename."""
        if self.filename:
            return "<Tile #{} - {}>".format(
                self.number, os.path.basename(self.filename)
            )
        return "<Tile #{}>".format(self.number)

def get_basename(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def calc_columns_rows(n):
    num_columns = int(ceil(sqrt(n)))
    num_rows = int(ceil(n / float(num_columns)))
    return (num_columns, num_rows)

def get_combined_size(tiles):
    # TODO: Refactor calculating layout to avoid repetition.
    columns, rows = calc_columns_rows(len(tiles))
    tile_size = tiles[0].image.size
    return (tile_size[0] * columns, tile_size[1] * rows)

def join(tiles):
    im = Image.new("RGBA", get_combined_size(tiles), None)
    for tile in tiles:
        try:
            im.paste(tile.image, tile.coords)
        except IOError:
            # do nothing, blank out the image
            continue
    return im

def validate_image(image, number_tiles):
    """Basic sanity checks prior to performing a split."""
    TILE_LIMIT = 99 * 99

    try:
        number_tiles = int(number_tiles)
    except BaseException:
        raise ValueError("number_tiles could not be cast to integer.")

    if number_tiles > TILE_LIMIT or number_tiles < 2:
        raise ValueError(
            "Number of tiles must be between 2 and {} (you \
                          asked for {}).".format(
                TILE_LIMIT, number_tiles
            )
        )

def validate_image_col_row(image, col, row):
    SPLIT_LIMIT = 99

    try:
        col = int(col)
        row = int(row)
    except BaseException:
        raise ValueError("columns and rows values could not be cast to integer.")

    if col < 1 or row < 1 or col > SPLIT_LIMIT or row > SPLIT_LIMIT:
        raise ValueError(
            f"Number of columns and rows must be between 1 and"
            f"{SPLIT_LIMIT} (you asked for rows: {row} and col: {col})."
        )
    if col == 1 and row == 1:
        raise ValueError("There is nothing to divide. You asked for the entire image.")

def img_cutter(filename, number_tiles=None, col=None, row=None, save=True):
    im = Image.open(filename)
    im_w, im_h = im.size

    columns = 0
    rows = 0
    if number_tiles:
        validate_image(im, number_tiles)
        columns, rows = calc_columns_rows(number_tiles)
    else:
        validate_image_col_row(im, col, row)
        columns = col
        rows = row

    tile_w, tile_h = int(floor(im_w / columns)), int(floor(im_h / rows))

    tiles = []
    number = 1
    for pos_y in range(0, im_h - rows, tile_h):  # -rows for rounding error.
        for pos_x in range(0, im_w - columns, tile_w):  # as above.
            area = (pos_x, pos_y, pos_x + tile_w, pos_y + tile_h)
            image = im.crop(area)
            position = (int(floor(pos_x / tile_w)) + 1, int(floor(pos_y / tile_h)) + 1)
            coords = (pos_x, pos_y)
            tile = Tile(image, number, position, coords)
            tiles.append(tile)
            number += 1
    if save:
        save_tiles(tiles, prefix=get_basename(filename), directory=os.path.dirname(filename))
    return tiles

def save_tiles(tiles, prefix="", directory=os.getcwd(), format="png"):
    for tile in tiles:
        tile.save(
            filename=tile.generate_filename(
                prefix=prefix, directory=directory, format=format
            ),
            format=format,
        )
    return tuple(tiles)

def reunion_image(tiles):
    image = join(tiles)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image


# ------------------ / Slice functions ------------------

# ------------------------ Utils ------------------------


def create_temp_dir(name_dir):
    if os.path.exists(name_dir): shutil.rmtree(name_dir)
    if not os.path.exists(name_dir): os.makedirs(name_dir)

def find_by_relative_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def adapt_image_to_show(image_to_prepare):
    old_image     = image_read(image_to_prepare)
    actual_width  = old_image.shape[1]
    actual_height = old_image.shape[0]

    if actual_width >= actual_height:
        max_val = actual_width
        max_photo_resolution = show_image_width
    else:
        max_val = actual_height
        max_photo_resolution = show_image_height

    if max_val >= max_photo_resolution:
        downscale_factor = max_val/max_photo_resolution
        new_width        = round(old_image.shape[1]/downscale_factor)
        new_height       = round(old_image.shape[0]/downscale_factor)
        resized_image    = cv2.resize(old_image,
                                   (new_width, new_height),
                                   interpolation = interpolation_mode)
        image_write("temp.png", resized_image)
        return "temp.png"
    else:
        new_width        = round(old_image.shape[1])
        new_height       = round(old_image.shape[0])
        resized_image    = cv2.resize(old_image,
                                   (new_width, new_height),
                                   interpolation = interpolation_mode)
        image_write("temp.png", resized_image)
        return "temp.png"

def prepare_output_filename(img, AI_model, target_file_extension):
    result_path = (img.replace("_resized" + target_file_extension, "").replace(target_file_extension, "") 
                    + "_"  + AI_model + target_file_extension)
    return result_path

def delete_list_of_files(list_to_delete):
    if len(list_to_delete) > 0:
        for to_delete in list_to_delete:
            if os.path.exists(to_delete):
                os.remove(to_delete)

def write_in_log_file(text_to_insert):
    log_file_name   = app_name + ".log"
    with open(log_file_name,'w') as log_file: log_file.write(text_to_insert) 
    log_file.close()

def read_log_file():
    log_file_name   = app_name + ".log"
    with open(log_file_name,'r') as log_file: step = log_file.readline()
    log_file.close()
    return step


# IMAGE

def image_write(path, image_data):
    _, file_extension = os.path.splitext(path)
    r, buff = cv2.imencode(file_extension, image_data)
    buff.tofile(path)

def image_read(image_to_prepare, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(image_to_prepare, dtype=np.uint8), flags)

def resize_image(image_path, resize_factor, target_file_extension):
    new_image_path = (os.path.splitext(image_path)[0] + "_resized" + target_file_extension).strip()

    old_image  = image_read(image_path.strip(), cv2.IMREAD_UNCHANGED)
    new_width  = int(old_image.shape[1] * resize_factor)
    new_height = int(old_image.shape[0] * resize_factor)

    resized_image = cv2.resize(old_image, (new_width, new_height), 
                                interpolation = interpolation_mode)    
    image_write(new_image_path, resized_image)

def resize_image_list(image_list, resize_factor, target_file_extension):
    files_to_delete   = []
    downscaled_images = []
    how_much_images = len(image_list)

    index = 1
    for image in image_list:
        resized_image_path = (os.path.splitext(image)[0] + "_resized" + target_file_extension).strip()
        
        resize_image(image.strip(), resize_factor, target_file_extension)
        write_in_log_file("Resizing image " + str(index) + "/" + str(how_much_images)) 

        downscaled_images.append(resized_image_path)
        files_to_delete.append(resized_image_path)

        index += 1

    return downscaled_images, files_to_delete


#VIDEO

def extract_frames_from_video(video_path):
    video_frames_list = []
    cap          = cv2.VideoCapture(video_path)
    frame_rate   = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    # extract frames
    video = VideoFileClip(video_path)
    img_sequence = app_name + "_temp" + os.sep + "frame_%01d" + '.jpg'
    video_frames_list = video.write_images_sequence(img_sequence, logger = 'bar', fps = frame_rate)
    
    # extract audio
    try:
        video.audio.write_audiofile(app_name + "_temp" + os.sep + "audio.mp3")
    except Exception as e:
        pass

    return video_frames_list

def video_reconstruction_by_frames(input_video_path, frames_upscaled_list, AI_model, cpu_number):
    cap          = cv2.VideoCapture(input_video_path)
    frame_rate   = int(cap.get(cv2.CAP_PROP_FPS))
    path_as_list = input_video_path.split("/")
    video_name   = str(path_as_list[-1])
    only_path    = input_video_path.replace(video_name, "")
    for video_type in supported_video_list: video_name = video_name.replace(video_type, "")
    upscaled_video_path = (only_path + video_name + "_" + AI_model + ".mp4")
    cap.release()

    audio_file = app_name + "_temp" + os.sep + "audio.mp3"

    clip = ImageSequenceClip.ImageSequenceClip(frames_upscaled_list, fps = frame_rate)
    if os.path.exists(audio_file):
        clip.write_videofile(upscaled_video_path,
                            audio   = audio_file,
                            threads = cpu_number)
    else:
        clip.write_videofile(upscaled_video_path,
                            threads = cpu_number)       

def resize_frame(image_path, new_width, new_height, target_file_extension):
    new_image_path = image_path.replace('.jpg', "" + target_file_extension)
    
    old_image = cv2.imread(image_path.strip(), cv2.IMREAD_UNCHANGED)

    resized_image = cv2.resize(old_image, (new_width, new_height), 
                                interpolation = interpolation_mode)    
    image_write(new_image_path, resized_image)

def resize_frame_list(image_list, resize_factor, target_file_extension, cpu_number):
    downscaled_images = []

    old_image = Image.open(image_list[1])
    new_width, new_height = old_image.size
    new_width = int(new_width * resize_factor)
    new_height = int(new_height * resize_factor)
    
    with ThreadPool(cpu_number) as pool:
        pool.starmap(resize_frame, zip(image_list, 
                                    itertools.repeat(new_width), 
                                    itertools.repeat(new_height), 
                                    itertools.repeat(target_file_extension)))

    for image in image_list:
        resized_image_path = image.replace('.jpg', "" + target_file_extension)
        downscaled_images.append(resized_image_path)

    return downscaled_images



# ----------------------- /Utils ------------------------


# ------------------ AI ------------------


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.sf = sf

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.sf == 4: self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.sf == 4: fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


# ------------------ /AI ------------------


# ----------------------- Core ------------------------


def thread_check_steps_for_images( not_used_var, not_used_var2 ):
    time.sleep(3)
    try:
        while True:
            step = read_log_file()
            if "Upscale completed" in step or "Error while upscaling" in step or "Stopped upscaling" in step:
                print(step)
                stop = 1 + "x"
            print(step)
            time.sleep(2)
    except:
        print("thread_check_steps_for_images")

def thread_check_steps_for_videos( not_used_var, not_used_var2 ):
    time.sleep(3)
    try:
        while True:
            step = read_log_file()
            if "Upscale video completed" in step or "Error while upscaling" in step or "Stopped upscaling" in step:
                print(step)
                stop = 1 + "x"
            print(step)
            time.sleep(2)
    except:
        print("thread_check_steps_for_videos")


def prepare_model(AI_model, device, half_precision):
    backend = torch.device(torch_directml.device(device))

    model_path = find_by_relative_path("AI" + os.sep + AI_model + ".pth")
    print("Loading model ", model_path)
    if "x2" in AI_model: upscale_factor = 2
    elif "x4" in AI_model: upscale_factor = 4

    model = RRDBNet(in_nc = 3, out_nc = 3, nf = 64, nb = 23, gc = 32, sf = upscale_factor)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    for _, v in model.named_parameters(): v.requires_grad = False
        
    if half_precision: model = model.half()
    model = model.to(backend)

    return model

def enhance(model, img, backend, half_precision):
    img = img.astype(np.float32)

    if np.max(img) > 256: max_range = 65535 # 16 bit images
    else: max_range = 255

    img = img / max_range
    if len(img.shape) == 2:  # gray image
        img_mode = 'L'
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA image with alpha channel
        img_mode = 'RGBA'
        alpha = img[:, :, 3]
        img = img[:, :, 0:3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
    else:
        img_mode = 'RGB'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ------------------- process image (without the alpha channel) ------------------- #
    
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    img = img.unsqueeze(0).to(backend)
    if half_precision: img = img.half()

    output = model(img) ## model
    
    output_img = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

    if img_mode == 'L':  output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

    # ------------------- process the alpha channel if necessary ------------------- #
    
    if img_mode == 'RGBA':
        alpha = torch.from_numpy(np.transpose(alpha, (2, 0, 1))).float()
        alpha = alpha.unsqueeze(0).to(backend)
        if half_precision: alpha = alpha.half()

        output_alpha = model(alpha) ## model

        output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
        output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)

        # merge the alpha channel
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
        output_img[:, :, 3] = output_alpha

    # ------------------------------ return ------------------------------ #
    if max_range == 65535: output = (output_img * 65535.0).round().astype(np.uint16) # 16-bit image
    else: output = (output_img * 255.0).round().astype(np.uint8)

    return output, img_mode



def reverse_split_multiple_frames(list_of_tiles_list, frames_upscaled_list):
    for index in range(len(frames_upscaled_list)):
        image_write(frames_upscaled_list[index], 
                    reunion_image(list_of_tiles_list[index]))              

def upscale_frame_and_save(frame, model, result_path, 
                            tiles_resolution, device, 
                            half_precision, list_of_tiles_list):

    used_tiles       = False
    backend          = torch.device(torch_directml.device(device))

    img_tmp          = image_read(frame)
    image_resolution = max(img_tmp.shape[1], img_tmp.shape[0])
    num_tiles        = image_resolution/tiles_resolution

    if num_tiles <= 1:
        with torch.no_grad():
            img_adapted     = image_read(frame, cv2.IMREAD_UNCHANGED)
            img_upscaled, _ = enhance(model, img_adapted, backend, half_precision)
            image_write(result_path, img_upscaled)
    else:
        used_tiles = True

        num_tiles  = round(num_tiles)
        if (num_tiles % 2) != 0: num_tiles += 1
        num_tiles  = round(num_tiles * multiplier_num_tiles)

        tiles = img_cutter(frame, num_tiles)
        with torch.no_grad():
            for tile in tiles:
                tile_adapted     = image_read(tile.filename, cv2.IMREAD_UNCHANGED)
                tile_upscaled, _ = enhance(model, tile_adapted, backend, half_precision)
                image_write(tile.filename, tile_upscaled)
                tile.image = Image.open(tile.filename)
                tile.coords = (tile.coords[0] * 4, 
                                tile.coords[1] * 4)

        list_of_tiles_list.append(tiles)

    return list_of_tiles_list, used_tiles

def process_upscale_video_frames(input_video_path, AI_model, resize_factor, device,
                                tiles_resolution, target_file_extension, cpu_number,
                                half_precision):
    try:
        start = timer()

        create_temp_dir(app_name + "_temp")

        write_in_log_file('...')
      
        write_in_log_file('Extracting video frames...')
        frame_list = extract_frames_from_video(input_video_path)
        
        if resize_factor != 1:
            write_in_log_file('Resizing video frames...')
            frame_list  = resize_frame_list(frame_list, 
                                            resize_factor, 
                                            target_file_extension, 
                                            cpu_number)

        write_in_log_file('Upscaling...')
        how_many_images = len(frame_list)
        done_images     = 0
        frames_upscaled_list = []
        list_of_tiles_list   = []

        model = prepare_model(AI_model, device, half_precision)

        for frame in frame_list:
            result_path = prepare_output_filename(frame, AI_model, target_file_extension)
            frames_upscaled_list.append(result_path)

            list_of_tiles_list, used_tiles = upscale_frame_and_save(frame, 
                                                                    model, 
                                                                    result_path, 
                                                                    tiles_resolution, 
                                                                    device, 
                                                                    half_precision,
                                                                    list_of_tiles_list)
            done_images += 1
            write_in_log_file("Upscaled frame " + str(done_images) + "/" + str(how_many_images))

        if used_tiles: 
            write_in_log_file("Reconstructing frames from tiles...")
            reverse_split_multiple_frames(list_of_tiles_list, frames_upscaled_list)

        write_in_log_file("Processing upscaled video...")
        video_reconstruction_by_frames(input_video_path, frames_upscaled_list, AI_model, cpu_number)

        write_in_log_file("Upscale video completed [" + str(round(timer() - start)) + " sec.]")

        create_temp_dir(app_name + "_temp")

    except Exception as e:
        write_in_log_file('Error while upscaling' + '\n\n' + str(e)) 
        import tkinter as tk
        error_root = tk.Tk()
        error_root.withdraw()
        tk.messagebox.showerror(title   = 'Error', 
                                message = 'Upscale failed caused by:\n\n' +
                                           str(e) + '\n\n' +
                                          'Please report the error on Github.com or Itch.io.' +
                                          '\n\nThank you :)')
        error_root.destroy()



def upscale_image_and_save(image, model, result_path, 
                            tiles_resolution, 
                            upscale_factor, 
                            device, half_precision):

    backend          = torch.device(torch_directml.device(device))

    original_image          = image_read(image)
    original_image_width    = original_image.shape[1]
    original_image_height   = original_image.shape[0]

    image_resolution = max(original_image_width, original_image_height)
    num_tiles        = image_resolution/tiles_resolution

    if num_tiles <= 1:
        with torch.no_grad():
            img_adapted     = image_read(image, cv2.IMREAD_UNCHANGED)
            img_upscaled, _ = enhance(model, img_adapted, backend, half_precision)
            image_write(result_path, img_upscaled)
    else:
        num_tiles = round(num_tiles)
        if (num_tiles % 2) != 0: num_tiles += 1
        num_tiles = round(num_tiles * multiplier_num_tiles)

        tiles = img_cutter(image, num_tiles)
        
        with torch.no_grad():
            for tile in tiles:
                tile_adapted     = image_read(tile.filename, cv2.IMREAD_UNCHANGED)
                tile_upscaled, _ = enhance(model, tile_adapted, backend, half_precision)
                image_write(tile.filename, tile_upscaled)
                tile.image = Image.open(tile.filename)
                tile.coords = (tile.coords[0] * upscale_factor, 
                                tile.coords[1] * upscale_factor)
    
        image_write(result_path, reunion_image(tiles))

        to_delete = []
        for tile in tiles: to_delete.append(tile.filename)
        delete_list_of_files(to_delete)

def process_upscale_multiple_images(image_list, AI_model, resize_factor, device, 
                                    tiles_resolution, target_file_extension, 
                                    half_precision):
    try:
        start = timer()
        write_in_log_file('...')

        if "x2" in AI_model: upscale_factor = 2
        elif "x4" in AI_model: upscale_factor = 4
        print(upscale_factor)        

        write_in_log_file('Resizing images...')
        image_list, files_to_delete = resize_image_list(image_list, resize_factor, target_file_extension)

        how_many_images = len(image_list)
        print("Number of images:",how_many_images,image_list )
        done_images     = 0

        write_in_log_file('Upscaling...')
        model = prepare_model(AI_model, device, half_precision)
        for img in image_list:
            result_path = prepare_output_filename(img, AI_model, target_file_extension)
            upscale_image_and_save(img, model, result_path, tiles_resolution, 
                                    upscale_factor, device, half_precision)
            done_images += 1
            write_in_log_file("Upscaled images " + str(done_images) + "/" + str(how_many_images))
                
        write_in_log_file("Upscale completed [" + str(round(timer() - start)) + " sec.]")

        delete_list_of_files(files_to_delete)
    except Exception as e:
        print("Image Error")
        write_in_log_file('Error while upscaling' + '\n\n' + str(e)) 
        import tkinter as tk
        error_root = tk.Tk()
        error_root.withdraw()
        tk.messagebox.showerror(title   = 'Error', 
                                message = 'Upscale failed caused by:\n\n' +
                                           str(e) + '\n\n' +
                                          'Please report the error on Github.com or Itch.io.' +
                                          '\n\nThank you :)')
        error_root.destroy()


# ----------------------- /Core ------------------------




#####
if __name__ == "__main__":
    multiprocessing.freeze_support()

    AI_model = "BSRGANx2"
    input_video_path = r"sample/2020110700707.mp4"
    input_video = True
    multi_img_list = ['sample/1.png','sample/2.png','sample/3.png']
    #multi_img_list = ['1.png','2.png','3.png']

    resize_factor = 100 #100% Resolution
    selected_resize_factor = 100#
    selected_VRAM_limiter = 2 # GB
    selected_cpu_number = 8 # Threads

    is_ready = True
    print("input_video_path",input_video_path)
    print("resize_factor",resize_factor)
    print("selected_resize_factor",selected_resize_factor)
    print("selected_VRAM_limiter",selected_VRAM_limiter, "GB")
    print("selected_cpu_number",selected_cpu_number)
    print("AI_model",AI_model)
    ### FILTER INPUTS 
    
    if compatible_gpus == 0:
        tk.messagebox.showerror(title   = 'Error', 
                                message = 'Sorry, your gpu is not compatible with QualityScaler :(')
        is_ready = False

    # resize factor
    try: resize_factor = int(float(str(selected_resize_factor)))
    except:
        print("Resize % must be a numeric value")
        is_ready = False

    #if resize_factor > 0 and resize_factor <= 100: resize_factor = resize_factor/100
    if resize_factor > 0: resize_factor = resize_factor/100
    else:
        print("Resize % must be a value > 0")
        is_ready = False
    
    # vram limiter
    try: tiles_resolution = 100 * int(float(str(selected_VRAM_limiter)))
    except:
        print("VRAM/RAM value must be a numeric value")
        is_ready = False 

    if tiles_resolution > 0: tiles_resolution = 100 * (vram_multiplier * int(float(str(selected_VRAM_limiter))))    
    else:
        print("VRAM/RAM value must be > 0")
        is_ready = False

    # cpu number
    try: cpu_number = int(float(str(selected_cpu_number)))
    except:
        print("Cpu number must be a numeric value")
        is_ready = False 

    if cpu_number <= 0:         
        print("Cpu number value must be > 0")
        is_ready = False
    elif cpu_number == 1: cpu_number = 1
    else: cpu_number = int(cpu_number)

    print("tiles_resolution",tiles_resolution)
    print("target_file_extension",target_file_extension)
    print("device",device)
    print("half_precision",half_precision)
    
    if not is_ready:
        print("Error with input variables exiting now")
        exit()
    else:    ##
        print("Ready")
        if input_video:
            print("Upscaling Video")
            target_file_extension=".mp4"
            
            process_upscale = multiprocessing.Process(target = process_upscale_video_frames,
                                                    args   = (input_video_path, 
                                                                AI_model, 
                                                                resize_factor, 
                                                                device,
                                                                tiles_resolution,
                                                                target_file_extension,
                                                                cpu_number,
                                                                half_precision))
            process_upscale.start()
            thread_wait = threading.Thread(target = thread_check_steps_for_videos,
                                        args   = (1, 2), 
                                        daemon = True)
            thread_wait.start()
            print("Upscale x4 Video")

        else:
            print("Processing Images")
            target_file_extension= ".png"
            process_upscale = multiprocessing.Process(target = process_upscale_multiple_images,
                                                        args   = (multi_img_list, 
                                                                AI_model, 
                                                                resize_factor, 
                                                                device,
                                                                tiles_resolution,
                                                                target_file_extension,
                                                                half_precision))
            
            process_upscale.start()
            thread_wait = threading.Thread(target = thread_check_steps_for_images,
                                            args   = (1, 2), daemon = True)
            thread_wait.start()
