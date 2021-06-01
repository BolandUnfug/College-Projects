/**
 *  Defines a visualization tool useful for a Susceptible-Infectious-Recovered (SIR) epidemiological model
 *  of viral spread, in the style of a cellular automaton (CA). <br>
 *  <br>
  *  @author Boland Unfug, Caitrin Eaton, spring 2021
 */

import javax.swing.*;
import java.awt.*;
import java.awt.image.*;

public class SIRScreen implements Screen {

    // Dimensions
    private  int cellSize = 1;           //  number of pixels along each side of each square cell within the window
    private final int mapSize;           //  number of cells along each side of the map
    private int windowSize;             // number of pixels on each side of the window
    private int windowWidth;             // pixel width of the window
    private int windowHeight;           // pixel height of the window
    private long frameTime = 500;        //  duration of each frame (day) in milliseconds

    // Graphics components
    private JFrame window;                 //  the window in which the model's animation is displayed
    private BufferedImage img;             //  the map of rectangles used to visualize all of the cells

    // SIR components
    private SIRMap map;                 //  the map of references to the actual cell objects themselves
    private SIRState[][] state;           //  a map in which each cell's state is represented as an enumerated value
    private char[][] type;               //  a map in which each cell's type is prepresented as a character

    // Cell colors: Basic (red), Masked (blue), Quarintine (yellow), SuperSafe (green), HalfMask(orange) Invalid type (light gray)
    private final static Color[] cellColor = { Color.RED, Color.BLUE, Color.YELLOW, Color.GREEN,Color.ORANGE, Color.LIGHT_GRAY};


    /**
     * SIRScreen constructor, creates an animated visualization for the given map of cells. <br>
     * @param world SIRMap, the 2D collection of cells that defines this particular SIR model
     */
    public SIRScreen(SIRMap world ) {
        // Determine the dimensions of components that will be visualized
        this.map = world;
        this.mapSize = world.getSize();
        this.cellSize = 1000/mapSize;
        
        this.windowSize = mapSize*cellSize;
        // Configure the graphics window
        this.window = new JFrame();
        this.img = new BufferedImage( this.windowSize, this.windowSize, BufferedImage.TYPE_INT_RGB );
        initWindow();
        update( 0 );
    }

    /**
     * Set the window's title string for the current frame of the animation
     * @param day (int) current day of the animation
     */
    private void setTitle( int day ){
        
        SIRState[][] demo = map.getPopulation();
        double s = 0;
        double i = 0;
        double r = 0;
        for (int row = 0; row < map.getSize(); row++){
            for(int col = 0; col < map.getSize(); col++){
                if(demo[row][col] == SIRState.SUSCEPTIBLE){
                    s++;
                }
                else if(demo[row][col] == SIRState.INFECTIOUS){
                    i++;
                }
                if(demo[row][col] == SIRState.RECOVERED){
                    r++;
                }
            }
        }
        s = s / (map.getSize()*map.getSize()) * 100;
        i = i / (map.getSize()*map.getSize()) * 100;
        r = r / (map.getSize()*map.getSize()) * 100;
        String title = String.format("Day %d, S: %.2f%%, I: %.2f%%, R: %.2f%%", day, s, i, r);
        this.window.setTitle( title );
    }

    /**
     * Initialize the graphics window and its contents.
     */
    private void initWindow() {

        // Configure the window itself
        window.setSize( this.windowSize+this.windowSize/50, this.windowSize+this.windowSize/25 );

        // Center the image in the graphics window
        ImageIcon icon = new ImageIcon( img );
        JLabel label = new JLabel( icon );
        window.add( label );

        // Make the graphics window visible
        window.setLocationRelativeTo(null);
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.setVisible(true);
        long initial_delay_ms = 2000;
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
    public void update( int day ) {
        // Update the current state map
        this.state = this.map.getPopulation();
        this.type = this.map.getPopulationTypes();
        // Use the window's title to track population demographics
        setTitle( day );

        // Configure the color of each cell according to its state: occupied or unoccupied
        state = map.getPopulation();
        Color typecolor = cellColor[4];
        int cell;
        for (int row = 0; row < mapSize; row++) {
            for (int col = 0; col < mapSize; col++ ){
                //find the color that represents this cells type
                if(type[row][col] != 'a'){
                    if (type[row][col] == 'u'){
                        typecolor = cellColor[0];
                    }
                    else if (type[row][col] == 'm'){
                        typecolor = cellColor[1];
                    }
                    else if (type[row][col] == 'q'){
                        typecolor = cellColor[2];
                    }
                    else if (type[row][col] == 's'){
                        typecolor = cellColor[3];
                    }
                }

            //Retrieving contents of a pixel
            cell = typecolor.getRGB();
            //Creating a Color object from pixel value
            Color color = new Color(cell, true);
            //Retrieving the R G B values
            int red = color.getRed();
            int green = color.getGreen();
            int blue = color.getBlue();
            //Modifying the RGB values
            //Creating new Color object



                // Find the color that reflects this cell's state
                if ( state[row][col] != null ) {
                    if(state[row][col] == SIRState.SUSCEPTIBLE){
                        int reddifferince = 255-color.getRed();
                        int greendifference = 255-color.getGreen();
                        int bluedifference = 255-color.getBlue();
                        red = reddifferince * 80/100 + color.getRed();
                        green = greendifference * 80/100  + color.getGreen();
                        blue = bluedifference * 80/100 + color.getBlue();
                        
                    }
                    else if(state[row][col] == SIRState.INFECTIOUS){
                        red = color.getRed();
                        green = color.getGreen();
                        blue = color.getBlue();
                    }
                    else if(state[row][col] == SIRState.RECOVERED){
                        red = color.getRed() -  color.getRed()* 80/100;
                        green = color.getGreen() - color.getGreen()* 80/100;
                        blue = color.getBlue() - color.getBlue()* 80/100;
                    }
                typecolor = new Color(red, green, blue);
                } else {
                    typecolor = cellColor[ SIRState.values().length ];  // Unrecognized state! Highlight in the ERROR color.
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

