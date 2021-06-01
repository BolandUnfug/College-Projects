/**
 * Compilation: javac MazeSolver_HelperFunctions.java
 * Execution: N/A
 *
 * This file is not meant to be run independently. Place this file in the same directory as your MazeSolver.java
 * and call these functions using the dot notation -- for example, MazeSolver_HelperFunctions.visualizeDistances() --
 * or copy and paste these functions into your MazeSolver.java file, citing the original author in their JavaDocs.
 *
 * Note that these functions also use ImageFilter.dispalyImage(), so it will be helpful to include that file in your
 * working director as well. Can you think of any other functions from Labs 01 - 03 that might come in handy?
 *
 * IMPORTANT:   If these functions are incompatible with your implementation, that's OK!
 *              You are welcome to modify these functions as you see fit!
 *              These specific implementations are not strict project requirements. They are provided only to
 *              help you reduce development & debugging time by taking care of some the heavy lifting that is useful
 *              for the program but not essential to Lab 04's primary learning objectives: 2D structures & objects.
 *
 * @author Caitirn Eaton
 */

import java.awt.*;
import java.awt.image.*;
import javax.imageio.*;
import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.lang.Math.*;
import java.util.ArrayList;
import java.util.LinkedList;
import java.time.YearMonth;
import java.awt.*;
import java.awt.image.*;
import javax.imageio.*;
import javax.swing.*;
import java.io.File;
import java.io.IOException;

public class MazeSolver_HelperFunctions{

    /**
     * Use colors to visualize each pixel's distance from the goal (red) with walls still drawn in black.
     * @param maze (BufferedImage) the original black and white maze image
     * @param dist (int[][]) in which each value represents the distance of a pixel from the goal
     */
    public static BufferedImage visualizeDistances( BufferedImage maze, int[][] dist, JFrame window ){

        //BufferedImage heatmap = ImageFilter.copy( maze );  // So that we don't alter the original maze

        int width = maze.getWidth();
        int height = maze.getHeight();
        int numPixels = width * height;
        int maxDist, red, green, blue, halfMax;
        Color rgb;

        // Find the largest distance in the map to help us calibrate the color gradient
        maxDist = 0;
        for( int y = 0; y < height; y++ ) {
            for (int x = 0; x < width; x++) {
                if ( maxDist < dist[ y ][ x ] ) {
                    maxDist = dist[ y ][ x ];
                }
            }
        }
        
        // Set the color of each pixel based on distance from the goal
        
        halfMax = maxDist / 2;
        for( int y = 0; y < height; y++ ) {
            for (int x = 0; x < width; x++) {
                if (dist[y][x] > 1) {
                    // Calculate RBG color channels as a function of distance from the goal pixel
                    red = (dist[y][x] - halfMax) * 255 / halfMax;
                    blue = 255 - Math.abs(halfMax - dist[y][x]) * 255 / halfMax;
                    green = (halfMax - dist[y][x]) * 255 / halfMax;

                    // Make sure color channels stay within the range [0, 255]
                    if (red < 0) red = 0;
                    else if (red > 255) red = 255;
                    if (green < 0) green = 0;
                    else if (green > 255) green = 255;
                    if (blue < 0) blue = 0;
                    else if (blue > 255) blue = 255;

                    // Apply the color to this pixel in the heatmap
                    rgb = new Color(red, green, blue);
                    maze.setRGB(x, y, rgb.getRGB());
                } else if (dist[y][x] < 1) {
                    // Empty space that is unreachable from the goal
                    maze.setRGB( x, y, Color.DARK_GRAY.getRGB() );
                } else {
                    // Wall
                    maze.setRGB( x, y, Color.BLACK.getRGB() );
                }
            }
        }
        // Make the heatmap appear on the user's screen
        ImageFilter.updateImage( maze, "Heatmap of distance, goal = red", window );
        return maze;
    }



    /**
     * Visualize the maze with the shortest path drawn in red.
     * @param maze (BufferedImage) the original black and white maze image
     * @param path (int[][]) in which each row is a {col, row} index pair representing the next step
     */
    public static BufferedImage visualizePath( BufferedImage maze, int[][] path, JFrame window ){

        if( path == null ) return null;

        //BufferedImage solution = ImageFilter.copy( maze );  // So that we don't alter the original maze

        // Recolor each pixel along the shortest path, in red
        int x, y;
        for( int step=0; step<path.length; step++ ){
            x = path[ step ][ 1 ];
            y = path[ step ][ 0 ];
            maze.setRGB( x, y, Color.RED.getRGB() );
        }

        // Make the solution image appear on the user's screen
        ImageFilter.updateImage( maze, "Shortest Path", window);
        return maze;
    }
}