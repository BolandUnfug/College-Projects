/**
 *  Defines an interface for a family of Map classes related to an SIR model. <br>
 *  <br>
 *  @author Boland Unfug, Caitrin Eaton, spring 2021
 */
public interface Map {

    /**
     * Accessor for map size.
     * @return size (int) the number of cells along each side of a square map.
     */
    int getSize();

    /**
     * Accessor for a copy of all of the cells' current states. Returns a map of States rather than the map of cells
     * in order to avoid exposing references to the cells themselves, while still providing up-to-date information about
     * the cell's current state.
     * @return the set of all cells' current states (SIRState[][])
     */
    SIRState[][] getPopulation();

    /**
     * Ask each cell within the map to update itself
     */
    void update(); 

}

