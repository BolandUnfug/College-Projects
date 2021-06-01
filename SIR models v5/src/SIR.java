/**
 *  Defines an interface for an SIR model. Each model should have a Map of Cells.
 *  Each Cell should have a current State and a next State defined by an enum which represents its cellular rules. <br>
 *  <br>
 *  @author Boland Unfug, Caitrin Eaton, spring 2021
 */
public interface SIR {
    /** Update the Map and Screen to the next generation */
    void update();

    /** Update the Map and Screen in a loop (calling update until a condition is met) */
    void simulate();
}
