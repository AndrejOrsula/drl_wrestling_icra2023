import cv2
import numpy as np
from controller import Camera, Robot


def _resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(
        image,
        dsize=(height, width),
        interpolation=cv2.INTER_LINEAR_EXACT,
    )


def _crop(
    image: np.ndarray, crop_left: int, crop_right: int, crop_top: int, crop_bottom: int
) -> np.ndarray:
    original_height, original_width = image.shape[:2]
    return image[
        crop_top : original_height - crop_bottom,
        crop_left : original_width - crop_right,
    ]


class __CameraSensor:
    def __init__(
        self,
        robot: Robot,
        time_step: float,
        device_name: str,
        crop_left: int = 0,
        crop_right: int = 0,
        crop_top: int = 0,
        crop_bottom: int = 0,
        resize_height: int = 0,
        resize_width: int = 0,
        redshift_color: bool = True,
    ):
        self.device: Camera = robot.getDevice(device_name)
        self.device.enable(time_step)
        self._camera_height = self.device.getHeight()
        self._camera_width = self.device.getWidth()
        self._height = self._camera_height
        self._width = self._camera_width

        self._crop_enable = (
            crop_left > 0 or crop_right > 0 or crop_top > 0 or crop_bottom > 0
        )
        if self._crop_enable:
            self._crop_left = crop_left
            self._crop_right = crop_right
            self._crop_top = crop_top
            self._crop_bottom = crop_bottom
            self._height -= crop_top + crop_bottom
            self._width -= crop_left + crop_right
        self._redshift_color = redshift_color

        if (resize_height > 0 or resize_width > 0) and (
            resize_height != self._height or resize_width != self.camera_top.width
        ):
            self.resize_enabled = True
            self._height = resize_height
            self._width = resize_width
        else:
            self.resize_enabled = False

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    def get_image_rgba(self) -> np.ndarray:
        img = np.frombuffer(self.device.getImage(), np.uint8).reshape(
            (self._camera_height, self._camera_width, 4)
        )
        if self._redshift_color:
            img[:, :, :3] = self.redshift(img[:, :, :3])
        if self._crop_enable:
            img = _crop(
                img,
                self._crop_left,
                self._crop_right,
                self._crop_top,
                self._crop_bottom,
            )
        if self.resize_enabled:
            img = _resize(img, self.height, self.width)

        return img

    def get_image_rgb(self) -> np.ndarray:
        img = np.frombuffer(self.device.getImage(), np.uint8).reshape(
            (self._camera_height, self._camera_width, 4)
        )[:, :, :3]
        if self._redshift_color:
            img = self.redshift(img)
        if self._crop_enable:
            img = _crop(
                img,
                self._crop_left,
                self._crop_right,
                self._crop_top,
                self._crop_bottom,
            )
        if self.resize_enabled:
            img = _resize(img, self.height, self.width)

        return img

    @staticmethod
    def redshift(
        image: np.ndarray,
        blue_hue: int = 111,
        hue_range: int = 10,
        min_saturation: int = 100,
        hue_shift: int = 65,
    ) -> np.ndarray:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_and(
            cv2.inRange(hsv_image[:, :, 0], blue_hue - hue_range, blue_hue + hue_range),
            cv2.inRange(hsv_image[:, :, 1], min_saturation, 255),
        )
        hsv_image[:, :, 0][mask > 0] += hue_shift
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


class CameraTop(__CameraSensor):
    def __init__(
        self,
        robot: Robot,
        time_step: float,
        device_name: str = "CameraTop",
        **kwargs,
    ):
        super().__init__(
            robot=robot,
            time_step=time_step,
            device_name=device_name,
            **kwargs,
        )


class CameraBottom(
    __CameraSensor,
):
    def __init__(
        self,
        robot: Robot,
        time_step: float,
        device_name: str = "CameraBottom",
        **kwargs,
    ):
        super().__init__(
            robot=robot,
            time_step=time_step,
            device_name=device_name,
            **kwargs,
        )


class CameraSensorsCombined:
    def __init__(
        self,
        robot: Robot,
        time_step: float,
        overlapping_pixels: int = 6,
        resize_height: int = 128,
        resize_width: int = 128,
    ):
        self.camera_top = CameraTop(
            robot=robot,
            time_step=time_step,
            crop_left=16,
            crop_right=16,
            crop_top=48,
            crop_bottom=10,
        )
        self.camera_bottom = CameraBottom(
            robot=robot,
            time_step=time_step,
            crop_left=16,
            crop_right=16,
            crop_top=4,
            crop_bottom=44,
        )
        self.overlapping_pixels = overlapping_pixels

        assert (
            self.camera_top.width == self.camera_bottom.width
        ), "Width of top and bottom camera must be equal to combine them."

        if (resize_height > 0 or resize_width > 0) and (
            resize_height
            != self.camera_top.height
            + self.camera_bottom.height
            - self.overlapping_pixels
            or resize_width != self.camera_top.width
        ):
            self.resize_enabled = True
            self._height = resize_height
            self._width = resize_width
        else:
            self.resize_enabled = False
            self._height = (
                self.camera_top.height
                + self.camera_bottom.height
                - self.overlapping_pixels
            )
            self._width = self.camera_top.width

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    def _combine_images(
        self, image_top: np.ndarray, image_bottom: np.ndarray
    ) -> np.ndarray:
        combined_image = np.empty(
            (
                image_top.shape[0] + image_bottom.shape[0] - self.overlapping_pixels,
                image_top.shape[1],
                image_top.shape[2],
            ),
            dtype=np.uint8,
        )
        combined_image[: image_top.shape[0] - self.overlapping_pixels] = image_top[
            : image_top.shape[0] - self.overlapping_pixels
        ]
        if self.overlapping_pixels > 0:
            combined_image[
                image_top.shape[0] - self.overlapping_pixels : image_top.shape[0]
            ] = (
                (
                    image_top[image_top.shape[0] - self.overlapping_pixels :].astype(
                        dtype=np.uint16
                    )
                    + image_bottom[: self.overlapping_pixels].astype(dtype=np.uint16)
                )
                // 2
            ).astype(
                dtype=np.uint8
            )
        combined_image[image_top.shape[0] :] = image_bottom[self.overlapping_pixels :]

        if self.resize_enabled:
            return _resize(combined_image, self.height, self.width)
        else:
            return combined_image

    def get_image_rgba(self) -> np.ndarray:
        return self._combine_images(
            image_top=self.camera_top.get_image_rgba(),
            image_bottom=self.camera_bottom.get_image_rgba(),
        )

    def get_image_rgb(self) -> np.ndarray:
        return self._combine_images(
            image_top=self.camera_top.get_image_rgb(),
            image_bottom=self.camera_bottom.get_image_rgb(),
        )
