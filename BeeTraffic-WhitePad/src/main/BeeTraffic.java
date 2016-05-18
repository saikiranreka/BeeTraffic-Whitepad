package main;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import opencv.Imageoperations;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import config.Configuration;

/**
 * @author saikiran
 * @version 1.2
 */
public class BeeTraffic {

	static {
		File file = new File("lib/libopencv_java244.so");
		System.load(file.getAbsolutePath());
	}

	static String target_path;
	static boolean debug = true;

	public static void main(String[] args) {
		Imageoperations imageop = new Imageoperations();
		target_path=Configuration.OUTPUT;
		if ((args.length == 2 && args[0].trim().equalsIgnoreCase("-dbg"))
				|| (args.length == 4 && args[2].trim().equalsIgnoreCase("-dbg"))) {
			if ((args.length == 2 && args[1].trim().equalsIgnoreCase("1"))
					|| (args.length == 4 && args[3].trim()
							.equalsIgnoreCase("1"))) {
				debug = true;
			}
		}

		if (args.length > 0 && args[0].trim().equalsIgnoreCase("-i")) {
			String[] split = args[1].trim().split("/");
			String name = split[split.length - 1].trim();
			processImage(imageop.readImage(args[1].trim()), name);
		} else {
			String input_path = Configuration.INPUT;
			Mat[] images = imageop.readImagesfromFolder(input_path);
			if (images == null) {
				return;
			}
			String[] names = imageop.getNames();

			for (int i = 0; i < images.length; i++) {
				processImage(images[i], names[i]);

			}
		}

	}

	/**
	 * @param image
	 *            - Mat. Image in mat
	 * @param name
	 *            - String. Name of the image.
	 */
	private static void processImage(Mat image, String name) {
		if (image.cols() == 0) {
			System.out.println("Didn't find the image in the path given");
			return;
		}
		Imageoperations imageop = new Imageoperations();
		image = image.submat(Configuration.ROW_START, Configuration.ROW_END,
				Configuration.COL_START, Configuration.COL_END);
		if (debug)
			imageop.writeImage(target_path + "/" + "crop_" + name, image);
		Mat copy_image = image.clone();

		// adjust brightness of the image
		image = adjustBrightness(image);

		// convert to HSV
		Mat hsv = new Mat();
		Imgproc.cvtColor(image, hsv, Imgproc.COLOR_BGR2HSV);

		// identify white color from hsv
		Mat dst2 = identifyWhite(hsv, name);
		
		//for debugging
		if (debug)
			imageop.writeImage(target_path + "/" + "white_" + name, dst2);
		

		// find the landing pad
		Rect rect = findLandingPad(dst2);

		// crop the landing pad
		image = image.submat(rect);

		if (debug)
			imageop.writeImage(target_path + "/" + "pad_" + name,
					image);
		
		copy_image = copy_image.submat(rect);
		copy_image = image.clone();

		Mat source = copy_image.clone();

		if (debug)
			imageop.writeImage(target_path + "/" + name, image);

		// Remove white background and Binarize the image
		source = removeBackground(image);
		if (debug)
			imageop.writeImage(target_path + "/" + "gray_" + name, source);

		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,
				new Size(2, 2));
		// Remove any noise
		for (int j = 0; j < 2; j++)
			Imgproc.dilate(source, source, element);

		for (int j = 0; j < 2; j++)
			Imgproc.erode(source, source, element);

		if (debug)
			imageop.writeImage(target_path + "/" + "erode_" + name, source);

		double totalArea = findBeeArea(source, copy_image, name);

		// System.out.println(names[i]);
//		 System.out.println(name + ","
//		 + Math.round(totalArea / Configuration.AVG_BEE_AREA));
		System.out.println(Math.round(totalArea / 100));
	}

	/**
	 * @param image
	 *            - Mat. Image in Mat to calculate the brightness
	 * @return brightness - double. Brightness of the image
	 */
	private static double getBrightness(Mat image) {
		Mat lum = new Mat();
		List<Mat> color = new ArrayList<Mat>();
		Core.split(image, color);

		Core.multiply(color.get(0), new Scalar(0.299), color.get(0));
		Core.multiply(color.get(1), new Scalar(0.587), color.get(1));
		Core.multiply(color.get(2), new Scalar(0.114), color.get(2));

		Core.add(color.get(0), color.get(1), lum);
		Core.add(lum, color.get(2), lum);

		Scalar summ = Core.sumElems(lum);

		double brightness = summ.val[0] / (image.rows() * image.cols() * 2);
		// System.out.println("brightness before "+ brightness);
		return brightness;
	}

	/**
	 * @param rotImg
	 *            - Mat. Image to rotate
	 * @param theta
	 *            - double. Angle to rotate
	 * @return rotatedImage - Result image after rotation.
	 */
	private static Mat RotateImage(Mat rotImg, double theta) {
		double angleToRot = theta;

		Mat rotatedImage = new Mat();
		if (angleToRot >= 92 && angleToRot <= 93) {
			Core.transpose(rotImg, rotatedImage);
		} else {
			org.opencv.core.Point center = new org.opencv.core.Point(
					rotImg.cols() / 2, rotImg.rows() / 2);
			Mat rotImage = Imgproc.getRotationMatrix2D(center, angleToRot, 1.0);

			Imgproc.warpAffine(rotImg, rotatedImage, rotImage, rotImg.size());
		}

		return rotatedImage;

	}

	/**
	 * @param image
	 *            - Mat to adjust the brightness
	 * @return image - Mat with adjusted brightness (80 <= brightness <=90)
	 */
	private static Mat adjustBrightness(Mat image) {
		double alpha = 1.5;
		double beta = 4;
		double brig = getBrightness(image);
		// System.out.println(brig);
		if (brig < 80) {
			beta = 80 - brig;
			image.convertTo(image, -1, alpha, beta);
		}
		if (brig > 90) {
			beta = brig - 200;
			image.convertTo(image, -1, 1.3, beta);
		}
		return image;
	}

	/**
	 * @param hsv
	 *            - Mat. Image convert to HSV
	 * @return dst1 - Mat. Image with white pixels representing red color.
	 */
	private static Mat identifyRed(Mat hsv) {
		Mat dst1 = new Mat();
		Mat dst = new Mat();

		Core.inRange(hsv, new Scalar(0, 80, 80), new Scalar(10, 255, 255), dst);
		Core.inRange(hsv, new Scalar(160, 80, 80), new Scalar(179, 255, 255),
				dst1);

		Core.add(dst, dst1, dst1);

		// erode and dialate
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS,
				new Size(2, 2));
		for (int j = 0; j < 5; j++)
			Imgproc.erode(dst1, dst1, element);
		for (int j = 0; j < 2; j++)
			Imgproc.dilate(dst1, dst1, element);

		return dst1;
	}

	/**
	 * @param hsv
	 *            - Mat. Image convert to HSV
	 * @return dst1 - Mat. Image with white pixels representing green color.
	 */
	private static Mat identifyGreen(Mat hsv) {
		Mat dst2 = new Mat();
		Core.inRange(hsv, new Scalar(35, 50, 50), new Scalar(90, 255, 255),
				dst2);

		// erode and dialate
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_DILATE,
				new Size(2, 2));

		for (int j = 0; j < 6; j++)
			Imgproc.erode(dst2, dst2, element);
		for (int j = 0; j < 3; j++)
			Imgproc.dilate(dst2, dst2, element);
		return dst2;
	}

	/**
	 * @param hsv
	 *            - Mat. Image convert to HSV
	 * @return dst1 - Mat. Image with white pixels representing white color.
	 */
	private static Mat identifyWhite(Mat hsv, String name) {
		Mat dst2 = new Mat();
		Mat dst1 = new Mat();
		Imageoperations imageop = new Imageoperations();
		// Identify red color
		Core.inRange(hsv, new Scalar(0, 50, 80), new Scalar(180, 255, 255),
				dst1);

		// Identify white and red Color
		Core.inRange(hsv, new Scalar(0, 0, 150), new Scalar(255, 255, 255),
				dst2);
		// subtract red from white and red
		Core.subtract(dst2, dst1, dst2);
		imageop.writeImage(Configuration.OUTPUT+"/hsv.png", hsv);
		imageop.writeImage(Configuration.OUTPUT+"/before_noise.png", dst2);
		// erode and dialate to remove noise
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_DILATE,
				new Size(3, 3));

		for (int j = 0; j < 8; j++)
			Imgproc.erode(dst2, dst2, element);
		for (int j = 0; j < 2; j++)
			Imgproc.dilate(dst2, dst2, element);
		return dst2;
	}

	/**
	 * @param dst2
	 *            - Mat
	 * @return angle - double. The skew angle of the image
	 */
	private static double findAngle(Mat dst2) {

		ArrayList<MatOfPoint> points = new ArrayList<MatOfPoint>();
		Imgproc.findContours(dst2, points, new Mat(), Imgproc.RETR_LIST,
				Imgproc.CHAIN_APPROX_NONE);

		// Rotate the image
		MatOfPoint2f point = new MatOfPoint2f();
		List<Point> list = new ArrayList<Point>();
		for (MatOfPoint contour : points) {
			list.addAll(contour.toList());
		}
		point.fromList(list);
		RotatedRect rrect = Imgproc.minAreaRect(point);
		double angle = rrect.angle;
		return angle;
	}

	/**
	 * @param dst2
	 *            - Mat, White color identified in the image
	 * @return Rect - Rect ROI
	 */
	private static Rect findLandingPad(Mat dst2) {
		ArrayList<MatOfPoint> points = new ArrayList<MatOfPoint>();
		Imgproc.findContours(dst2, points, new Mat(), Imgproc.RETR_LIST,
				Imgproc.CHAIN_APPROX_NONE);

		// Sort the contours on increasing row number
		Collections.sort(points, new Comparator<MatOfPoint>() {
			@Override
			public int compare(MatOfPoint o1, MatOfPoint o2) {
				Rect rect1 = Imgproc.boundingRect(o1);
				Rect rect2 = Imgproc.boundingRect(o2);
				if (rect1.y >= rect2.y) {
					return 1;
				} else {
					return -1;
				}
			}
		});

		// add contours to find bounding rectangle.
		double avg = 0;
		int count = 0;
		for (MatOfPoint contour : points) {
			Rect rect = Imgproc.boundingRect(contour);
			double area = Imgproc.contourArea(contour);
			if (area > 3530) {
				avg = rect.y;
				count = 1;
				break;
			}
			if (area > 20) {
				count++;
				avg += rect.y;
			}
		}

		avg = avg / count;
		MatOfPoint allPoints = new MatOfPoint();
		for (MatOfPoint contour : points) {
			Rect rect = Imgproc.boundingRect(contour);
			if (rect.y > avg - 23 && rect.y < avg + 23) {
				allPoints.push_back(contour);
			}
		}
		Rect rect = Imgproc.boundingRect(allPoints);

		// If the landing pad area is more than 12000, the identified pad need
		// more processing and filtering.
		if (rect.area() > 12000) {
			avg = 0;
			int change = 25;
			for (MatOfPoint contour : points) {
				Rect temprect = Imgproc.boundingRect(contour);
				double area = Imgproc.contourArea(contour);
				if (area > 4000) {
					avg = temprect.y;
					count = 1;
					change = 7;
					break;
				}
				if (area > 20) {
					count++;
					avg += temprect.y;
				}
			}
			avg = avg / count;
			allPoints = new MatOfPoint();
			for (MatOfPoint contour : points) {
				double area = Imgproc.contourArea(contour);
				Rect temprect = Imgproc.boundingRect(contour);
				if (temprect.y > avg - change && temprect.y < avg + change
						&& area > 63) {
					allPoints.push_back(contour);
				}
			}
			rect = Imgproc.boundingRect(allPoints);
		}
		// Rect newRect = new Rect(rect.x, rect.y, rect.height + 6, rect.width);
		return rect;
	}

	/**
	 * @param image
	 *            - Mat representing the white landing pad
	 * @return source - Mat with white background removed and black pixels
	 *         representing bees
	 */
	private static Mat removeBackground(Mat image) {
		Mat source = image.clone();
		for (int row = 0; row < image.rows(); row++) {
			for (int col = 0; col < image.cols(); col++) {
				double[] pixel = image.get(row, col);
				if (row <= 3 || col <= 3 || row >= image.rows() - 4
						|| col >= image.cols() - 3) {
					pixel[0] = 255;
					pixel[1] = 255;
					pixel[2] = 255;
				} else if (pixel[0] >= 120 && pixel[1] >= 150
						&& pixel[2] >= 180) {
					pixel[0] = 255;
					pixel[1] = 255;
					pixel[2] = 255;
				} else {
					pixel[0] = 0;
					pixel[1] = 0;
					pixel[2] = 0;

				}
				source.put(row, col, pixel);
			}
		}
		return source;
	}

	/**
	 * @param source
	 *            - Mat. Image after removing white background and noise
	 * @param image
	 *            - Mat. Original Image with landing pad to draw contours
	 * @param name
	 *            - String. Name of the image
	 * @return area - double. Total area of the bees in the image
	 */
	private static double findBeeArea(Mat source, Mat image, String name) {
		Imageoperations imageop = new Imageoperations();
		Mat bImage = image.clone();
		Mat grayImage = image.clone();
		ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		List<MatOfPoint> refinedContours = new ArrayList<MatOfPoint>();

		// Convert to gray
		Imgproc.cvtColor(source, grayImage, Imgproc.COLOR_RGB2GRAY);

		// Find contours. Each Contour represents a bee or group of bees
		Imgproc.findContours(grayImage, contours, new Mat(), Imgproc.RETR_LIST,
				Imgproc.CHAIN_APPROX_TC89_L1);
		Imgproc.drawContours(bImage, contours, -1, new Scalar(0, 0, 255));
		double totalArea = 0;
		for (MatOfPoint contour : contours) {
			double area = Imgproc.contourArea(contour);
			// Check if the contour is valid group of bees of noise
			if (area > 20 && area < 3000) {
				totalArea += area;
				refinedContours.add(contour);
			}
		}
		Imgproc.drawContours(image, refinedContours, -1, new Scalar(0, 0, 255));
		if (debug)
			imageop.writeImage(target_path + "/res_" + name, image);
		return totalArea;
	}
}
