/******************************************************************************
 *  Compilation:  javac ImageIntro.java
 *  Execution:    java ImageIntro [filePath (String)]
 *
 *  Displays an image using Java's Swing & AWT (Abstract Window Toolkit) packages.
 *  In Tuesday's class, we'll talk through Image objects, copying an Image pixel by pixel,
 *  and creating a swapRedBlue filter that manipulates pixels.
 *
 *  % java ImageFilter image.png
 *      Displays image.png and copies it into image_copy.png. Hopefully, by the end of class,
 *      this will also apply a swapRedBlue filter to image.png and save the result in image_swapRedBlue.png
 *
 * @author YOUR NAME HERE
 * @author Caitrin Eaton
 *
 ******************************************************************************/
import java.awt.*;
import java.awt.image.*;
import javax.imageio.*;
import javax.swing.*;
import java.io.File;
import java.io.IOException;

public class ImageFilter {

    /**
     * Copy the original image's pixels into the output image.
     * @param original (BufferedImage) the original image
     * @return duplicate (BufferedImage) a new copy of the original image
     */
    public static BufferedImage copy( BufferedImage original ){
        // The output image begins as a blank image that is the same size and type as the original
        BufferedImage duplicate = new BufferedImage(original.getWidth(), original.getHeight(), original.getType());

        // Iterate over the original, copying each pixel's RGB color over to the new image (the copy)
        int rgb, red, green, blue;
        Color colorIn, colorOut;
        for (int row=0; row<original.getHeight(); row++){
            for (int col=0; col<original.getWidth(); col++){
                rgb = original.getRGB( col, row );
                // Casting the RGB integer to a Color is unnecessary in this case, but has been included here
                // as an example of how you can access the red (R), green (G), and blue (B) channels individually,
                // which will be essential for creating your own filters in the future.
                colorIn = new Color( rgb );
                red = colorIn.getRed();
                green = colorIn.getGreen();
                blue = colorIn.getBlue();
                colorOut = new Color( red, green, blue );
                //System.out.println( String.format("original[%d][%d] = %d, %d, %d", row, col, red, green, blue) );
                duplicate.setRGB( col, row, colorOut.getRGB() );
            }
        }

        // Return a reference to the shiny new copy of the input image
        return duplicate;
    }


    public static BufferedImage resizeImage(BufferedImage originalImage) {
        Dimension size = Toolkit.getDefaultToolkit().getScreenSize(); 
        //System.out.println(size);
        int targetWidth = (int)size.getWidth();
        int targetHeight = (int)size.getHeight();
        //System.out.println(originalImage.getWidth());
        //System.out.println(originalImage.getHeight());
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_BYTE_BINARY);
        //System.out.println(resizedImage.getWidth());
        //System.out.println(resizedImage.getHeight());
        //graphics2D.dispose();
        return resizedImage;
    }

    /**
     * Swap the red and blue color channels in the original image's pixels.
     * @param original (BufferedImage) the original image
     * @return filtered (BufferedImage) the filtered image
     */
    public static BufferedImage swapRedBlue( BufferedImage original ){
        // The output image begins as a blank image that is the same size and type as the original
        BufferedImage filtered = null; // hmm... that doesn't seem right

        // Iterate over the original, swapping the red and blue color channels of each pixel
        // Hint: start with something similar to copy(), and modify from there.

        // Return a reference to the shiny new filtered image
        return filtered;
    }

    /**
     * Make the image visible on the screen, in its own window.
     * @param img (BufferedImage) the image to display
     * @param title (String) the title and caption of the image
     * @return JFrame in which the image is displayed
     */
    public static JFrame displayImage( BufferedImage img, String title ){
        // Create the graphics window
        JFrame window = new JFrame();
        window.setTitle( title );
        resizeImage(img);
        window.setSize( img.getWidth()+20, img.getHeight()+40 );
        //System.out.println(img.getWidth()+20 );
        //System.out.println(img.getHeight()+20);
        // Center the image in the graphics window
        ImageIcon icon = new ImageIcon( img );
        JLabel label = new JLabel( icon );
        window.add( label );

        // Make the graphics window visible until the user closes it (which also ends the program)
        window.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE); //JFrame.EXIT_ON_CLOSE);
        window.setVisible(false);

        // Return a reference to the display window, so that we can manipulate it in the future, if we like.
        return window;
    }
    public static JFrame updateImage( BufferedImage img, String title, JFrame window){

        ImageIcon icon = new ImageIcon( img );
        JLabel label = new JLabel( icon );
        window.add( label );

        // Make the graphics window visible until the user closes it (which also ends the program)
        window.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE); //JFrame.EXIT_ON_CLOSE);
        window.setVisible(true);
        return window;
    }
    /**
     * Open the source image, display it, copy it, swapRedBlue filter it, and save the output images.
     * @param inFileName (String) the path to the input file
     */
    public static boolean filterImage( String inFileName ){

        // Read in the original image from the input file
        BufferedImage original = null;
        try {
            original = ImageIO.read(new File(inFileName));
        } catch (IOException e) {
            System.err.println( String.format("%s%n", e) );
            return false;
        }
        displayImage( original, inFileName );

        /*
        // Copy the image
        BufferedImage copied;
        copied = copy(original);
        displayImage( copied, "Copy" );
        */

        /*
        // Save the copy in a new image file
        int period = inFileName.indexOf( "." );
        String fileExtension = inFileName.substring( period+1 );
        String copyFileName = inFileName.substring( 0, period ) + "_copy." + fileExtension;
        try {
            File copiedFile = new File( copyFileName );
            ImageIO.write( copied, fileExtension,  copiedFile );
        } catch (IOException e) {
            System.err.println( String.format("%s%n", e) );
            return false;
        }
        */

        // Swap the image's red and blue color channels
        // BufferedImage filtered = swapRedBlue( original );

        // Save the filtered image in a new image file
        // String filteredFileName = inFileName.substring( 0, period ) + "_swapRedBlue." + fileExtension;
        // ... then what? ...

        return true;    // Success!
    }

    /**
     * Parses commandline arguments, then triggers the filter.
     * @param args (String[]) commandline arguments: file path, filter name, and filter parameter (if applicable)
     */
    public static void main(String[] args) {

        // If the user misses a commandline argument, show them a helpful usage statement
        String usageStatement = "USAGE: java ImageFilter filePath"
                + "\nFor example:"
                + "\n\tjava ImageFilter image.png"
                + "The image's file extension must be PNG, JPEG, or JPG.";

        // Parse commandline arguments
        String fileName = "";
        if( args.length > 0 ){
            fileName = args[0];
        } else {
            System.out.println( usageStatement );
            return;
        }

        // Open the source image, filter it, and save the output image
        boolean successful = filterImage( fileName );
        if (!successful) {
            System.out.println( "ERROR: Failed to filter the image." );
            System.out.println( usageStatement );
        }

    }
}
