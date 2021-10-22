/**
 *  Defines a visualization tool useful for a Susceptible-Infectious-Recovered (SIR) epidemiological model
 *  of viral spread, in the style of a cellular automaton (CA). <br>
 *  <br>
  *  @author Boland Unfug, Caitrin Eaton, spring 2021
 */

import javax.swing.*;
import java.awt.*;
import java.awt.image.*;

public class ParityVisualized implements Visualizer {

    // Dimensions
    private  int cellSize = 10;           //  number of pixels along each side of each square cell within the window
    private final int mapSize;           //  number of cells along each side of the map
    private int windowSize;             // number of pixels on each side of the window
    private int windowWidth;             // pixel width of the window
    private int windowHeight;           // pixel height of the window
    private long frameTime = 3000;        //  duration of each frame (day) in milliseconds

    // Graphics components
    private JFrame window;                 //  the window in which the model's animation is displayed
    private BufferedImage img;             //  the map of rectangles used to visualize all of the cells

    // SIR components
    private ParityMap bitMap;                 //  the map of references to the actual cell objects themselves
    private int[][] bitMatrix;
    
    // Colors: White = on, Black = off
    private final static Color[] cellColor = { Color.WHITE, Color.BLACK, Color.GREEN, Color.RED, Color.GRAY};


    /**
     * SIRScreen constructor, creates an animated visualization for the given map of cells. <br>
     * @param world SIRMap, the 2D collection of cells that defines this particular SIR model
     * @return 
     */
    public ParityVisualized( ParityMap bitMap) {
        // Determine the dimensions of components that will be visualized
        this.bitMap = bitMap;
        this.mapSize = this.bitMap.getSize();
        this.cellSize = 512/mapSize;
        
        this.windowSize = mapSize*cellSize;
        // Configure the graphics window
        this.window = new JFrame();
        this.img = new BufferedImage( this.windowSize, this.windowSize, BufferedImage.TYPE_INT_RGB );
        initWindow();
        update();
    }

    /**
     * Initialize the graphics window and its contents.
     */
    private void initWindow() {

        // Configure the window itself
        window.setSize( this.windowSize*2, this.windowSize*2 );

        // Center the image in the graphics window
        ImageIcon icon = new ImageIcon( img );
        JLabel label = new JLabel( icon );
        window.add( label );

        // Make the graphics window visible
        window.setLocationRelativeTo(null);
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.setVisible(true);
        long initial_delay_ms = 1000;
        window.repaint( initial_delay_ms );

        // Hold the initial population steady for 2 seconds before starting the simulation
        try {
            Thread.sleep(initial_delay_ms);
        } catch (InterruptedException e) { }
    }

    /**
     * Control the speed of the animation. <br>
     * @param milliseconds long, the duration of a single frame, in milliseconds
     */
    public void setFrameTime(long milliseconds ){
        this.frameTime = milliseconds;
        System.out.println(frameTime);
    }

    /**
     * Control the speed of the animation automatically. <br>
     * @param milliseconds long, the duration of a single frame, in milliseconds
     */
    public void setFrameTime(){
        this.frameTime = 0;
        if(mapSize <=500){
            this.frameTime = 10000/mapSize;
        }
        else{
            this.frameTime = 1000/mapSize;
        }
        System.out.println(frameTime);
    }

    /**
     * Update the graphical representation of the map to reflect the map's current state.
     * @param day (int) current day in the simulation
     */
    public void update() {
        // Update the current state map
        this.bitMatrix = this.bitMap.getMatrix();
        Color typecolor = cellColor[4];
        for (int row = 0; row < mapSize; row++) {
            for (int col = 0; col < mapSize; col++ ){
                //find the color that represents this cells type
                if(bitMatrix[row][col] == 0){
                   typecolor = cellColor[0];
                }
                else if(bitMatrix[row][col] == 2 || bitMatrix[row][col] == 3){
                    typecolor = cellColor[3];
                }
                else{
                    typecolor = cellColor[1];
                }
                
                // in case a cell has more than 1 pixel along each edge
                for( int dr=0; dr<cellSize; dr++) {
                    for (int dc = 0; dc < cellSize; dc++) {
                        img.setRGB( col*cellSize+dc, row*cellSize+dr, typecolor.getRGB() );
                    }
                }
            }
        }

        window.repaint( frameTime );
        try {
            Thread.sleep( frameTime );
        } catch (InterruptedException e) { }
    }
}

