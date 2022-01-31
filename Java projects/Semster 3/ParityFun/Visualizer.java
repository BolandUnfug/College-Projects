/**
 *  Defines an interface for SIR model visualization tools. <br>
 *  <br>
 *  @author Boland Unfug, spring 2021
 */
public interface Visualizer {
    /**
    * update the graphical representation of the grid to reflect each cell's current state
    */
    void update();

    /**
     * control the speed of animation
     * @param milliseconds (long) the length of time (in milliseconds) for which each frame will be displayed
     */
    void setFrameTime( long milliseconds );

    /**
     * Something to interact with the screen I guess. to be added later.
     */
}
