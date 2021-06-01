/**
 *  Defines the cell class for several subtype cell classes: object-oriented implementation <br>
 * <br>
 *  @author Boland Unfug, Caitrin Eaton, spring 2021
 */
import java.util.Random;

public class SIRCell implements Cell {

    // Possible states common to all Cells
    public final static int NEIGHBORHOOD_SIZE = 8;

    // Attributes particular to this Cell
    protected Random rand = new Random();
    protected SIRState currentState;
    protected SIRState nextState;
    protected int totalNeighbors = 0;
    protected int infectedNeighbors = 0;
    protected int susceptableNeighbors = 0;
    protected int recoveredNeighbors = 0;
    protected SIRCell[] neighborhood;
    protected double infectionRate = .166;
    protected double recoveryRate = .0366;
    protected char celltype;
    /**
     * Construct a new cell for an SIR model
     * @param initState SIRState, whether the cell is created SUSCEPTIBLE or INFECTED
     */
    public SIRCell( SIRState initState ) {
        this.currentState = initState;
        this.nextState = this.currentState;
        this.totalNeighbors = 0;
        this.infectedNeighbors = 0;
        this.celltype = 'b';
        this.neighborhood = new SIRCell[ NEIGHBORHOOD_SIZE ]; // references are initially null
    }
    public SIRCell(){

    }

    /**
     * Accessor for a cell's current state
     * @return currentState SIRState, whether the cell is currently SUSCEPTIBLE or INFECTED
     */
    public SIRState getCurrentState(){
        return this.currentState;
    }

    /**
     * Accessor for a cell's next state
     * @return nextState SIRState, whether the cell is currently SUSCEPTIBLE or INFECTED
     */
    public SIRState getNextState(){
        return this.nextState;
    }

    public char getType(){
        return this.celltype;
    }

    /**
     * Accessor for a cell's infected neighbors
     * @return infectedNeighbors int, how many neighbors the cell believes are currently Infected
     */
    public int getInfectedNeighbors(){
        return this.infectedNeighbors;
    }

    /**
     * Accessor for a cell's susceptable neighbors
     * @return susceptableNeighbors int, how many neighbors the cell believes are currently Susceptable
     */
    public int getSusceptableNeighbors(){
        return this.susceptableNeighbors;
    }

    /**
     * Accessor for a cell's recovered neighbors
     * @return recoveredNeighbors int, how many neighbors the cell believes are currently Recovered
     */
    public int getRecoveredNeighbors(){
        return this.recoveredNeighbors;
    }

    /**
     * Accessor for a cell's infection rate
     * @return infectionRate double, how likely the cell is to spread the virus
     */
    public double getInfectionRate(){
        return this.infectionRate;
    }

    /**
     * Accessor for a cell's recovery rate
     * @return recoveryRate double, how likely the cell is to recover
     */
    public double getRecoveryRate(){
        return this.recoveryRate;
    }
    
    /**
     * adds a cell to this cell's neighborhood
     * @param cell SIRCell, a reference to a neighboring cell object
     */
    public void addNeighbor(Cell cell) {
        // Make sure there's room in the neighborhood
        if (this.totalNeighbors < NEIGHBORHOOD_SIZE) {
            // Add the new neighbor to the neighborhood
            this.neighborhood[totalNeighbors] = (SIRCell) cell;
            this.totalNeighbors++;
            // Keep the living neighbors counter up to date
        if (cell != null){
            if (cell.getCurrentState() == SIRState.SUSCEPTIBLE) {
                this.susceptableNeighbors++;
            }
        }
        } else{
            System.out.println("Warning: new neighbor not added; neighborhood full.");
        }
    }

    /**
     * Updates a cell's next state according to the cells infection rate.
     */
    public void updateNextState(){
        if(this.currentState == SIRState.SUSCEPTIBLE){
            for(int i = 0; i<this.totalNeighbors; i++ ){
                if (this.neighborhood[i].getCurrentState() == SIRState.INFECTIOUS ) {
                    if(rand.nextDouble() < neighborhood[i].getInfectionRate()){
                        this.nextState = SIRState.INFECTIOUS;
                    }
                }
            }
        }
        else if(this.currentState == SIRState.INFECTIOUS){
                if(rand.nextDouble() < getRecoveryRate()){
                        this.nextState = SIRState.RECOVERED;
                    }
        }
    }
    /**
     * Updates a cell's current state using a next state previously determined by updateNextState()
     */
    public void updateCurrentState(){
        this.currentState = this.nextState;
    }
}
