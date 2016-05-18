package opencv;

import java.io.File;
import java.util.Arrays;
import java.util.Comparator;

import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

public class Imageoperations {

	static String[] names;

	public Mat readImage(String path) {
		return Highgui.imread(path);
	}

	public Mat[] readImagesfromFolder(String folderpath) {
		File folder = new File(folderpath);
		File[] images = folder.listFiles();
		if(images==null||images.length==0){
			System.out.println("Didn't find the given input folder");
			return null;
		}
		Arrays.sort(images, new Comparator<File>() {
			public int compare(File a, File b) {
				return a.getName().compareTo(b.getName());
			}
		});
		Mat[] sources = new Mat[images.length];
		names = new String[images.length];
		for (int i = 0; i < images.length; i++) {
			sources[i] = readImage(images[i].getAbsolutePath());
			names[i] = images[i].getName();
			System.out.println(names[i]);
		}
		return sources;
	}

	public String[] getNames() {
		return names;
	}

	public void writeImage(String path, Mat img) {
		Highgui.imwrite(path, img);
	}
}
