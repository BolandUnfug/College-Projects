/**
 *  Defines an interface for a family of cell classes. <br>
 *  <br>
 *  @author Boland Unfug, Caitrin Eaton, spring 2021
 */
public interface Cell {
    
    /** All neighborhoods must have 8 neighbors */
    public final int NEIGHBORHOOD_SIZE = 8;

    // Methods that all Cells must implement, regardless of Cell Type

    /**
     * Add another cell to this cell's neighborhood
     * @param cell (Cell): a reference to a neighboring Cell
     */
    void addNeighbor( Cell cell );

    /**
     * Accessor for a cell's current state
     * @return CurrentState (eg. State.INFECTIOUS)
     */
    SIRState getCurrentState();

    /**
     * Accessor for a cell's next state
     * @return NextState (e.g. eg. State.INFECTIOUS)
     */
    SIRState getNextState();

    /**
     * Update a cell's current state to match its (previously decided) next state.
     */
    void updateCurrentState();

    /**
     * Apply the cell's rules in order to determine this cell's next state.
     */
    void updateNextState();
}
