/**
 *  Defines an interface for SIR model visualization tools. <br>
 *  <br>
 *  @author Boland Unfug, Caitrin Eaton, spring 2021
 */
public interface Screen {
    /**
    * update the graphical representation of the grid to reflect each cell's current state
    * @param timeStamp (int) to display in the window's title bar (e.g. the current generation, in a Game of Life)
    */
    void update( int timeStamp );

    /**
     * control the speed of animation
     * @param milliseconds (long) the length of time (in milliseconds) for which each frame will be displayed
     */
    void setFrameTime( long milliseconds );
}
