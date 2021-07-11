/**
 *  Defines the map used in the SIR model: object-oriented implementation <br>
 * <br>
 *  @author Boland Unfug, Caitrin Eaton, spring 2021
 */

import java.util.Random;

public class SIRMap implements Map {
    
    // State-related map attributes
    protected int size;
    protected SIRCell[][] population;
    protected boolean probability;

    /**
     * Construct a new map for the SIR model.
     * @param size int, the width (and height) of the map, in cells
     */
    public SIRMap(int size) {
        this.size = size;
        this.population = new SIRCell[size][size]; // references are initially null
    }

    public SIRMap() {
        this.size = 1;
    }


    /**
     * Read the size of the square population array.
     * @return size int, the number of cells along the width (and height) of the square population array.
     */
    public int getSize() {
        return this.size;
    }

    /**
     * Returns if the map is using kernel distributions or not.
     * @return boolean probability, if its kernel distributions or not.
     */
    public boolean getProbability() {
        return this.probability;
    }

    /**
     * Sets if the map is using kernel distributions or not.
     * @return boolean probability, if its kernel distributions or not.
     */
    public void setProbability() {
        probability = true;
    }

    /**
     * Read the amount of infected cells.
     * @return infected int, the number of infected cells.
     */
    public int getInfected(){
        int i = 0;
        for (int row = 0; row < this.size; row++){
            for(int col = 0; col < this.size; col++){
                if(this.population[row][col].getCurrentState() == SIRState.INFECTIOUS){
                    i++;
                }
            }
        }
        return i;
    }

    /**
    * Read the amount of susceptable cells.
    * @return susceptable int, the number of susceptable cells.
    */
    public int getSusceptable(){
        int s = 0;
        for (int row = 0; row < this.size; row++){
            for(int col = 0; col < this.size; col++){
                if(this.population[row][col].getCurrentState() == SIRState.SUSCEPTIBLE){
                    s++;
                }
            }
        }
        return s;
    }

    /**
     * Read the amount of recovered cells.
     * @return recovered int, the number of recovered cells.
     */
    public int getRecovered(){
        int r = 0;
        for (int row = 0; row < this.size; row++){
            for(int col = 0; col < this.size; col++){
                if(this.population[row][col].getCurrentState() == SIRState.RECOVERED){
                    r++;
                }
            }
        }
        return r;
    }

    /**
     * Randomly initialize the map's population.
     * @param density double, the percent of occupied cells
     * @param maskdensity double, the percent of occupied cells
     */
    public void populate(double density, double maskdensity) {
        Random occupier = new Random();
        SIRState startingstate = SIRState.SUSCEPTIBLE;
        for (int row = 0; row < this.size; row++) {
            for (int col = 0; col < this.size; col++) {
                double occupancy = occupier.nextDouble();
                if (occupancy < density) {
                    startingstate = SIRState.INFECTIOUS;
                }
                else{
                    startingstate = SIRState.SUSCEPTIBLE;
                }
                double masks = occupier.nextDouble();
                if(masks < maskdensity){
                    masks = occupier.nextDouble();
                    if(masks < .33){
                        this.population[row][col] = new MaskedSIRCell(startingstate);
                    }
                    else if (masks < .66){
                        this.population[row][col] = new QuarintineSIRCell(startingstate);
                    }
                    else{
                        this.population[row][col] = new SuperSafeSIRCell(startingstate);
                    }
                }
                else{
                    this.population[row][col] = new UnMaskedSIRCell(startingstate);
                }
                
            }
        }
        formNeighborhoods();
    }

    /**
     * Connect cells to their neighbors.
     */
    protected void formNeighborhoods() {

        // Each cell in the map has its own neighborhood
        for (int row = 0; row < this.size; row++) {
            for (int col = 0; col < this.size; col++) {

                // Add this cell's 8 Moore neighbors to its neighborhood.
                for (int dr = -1; dr < 2; dr++) {
                    for (int dc = -1; dc < 2; dc++) {

                        // A cell shouldn't be its own neighbor.
                        if (dr != 0 || dc != 0) {
                            // Watch out for out of bounds exceptions.
                            if ((row + dr >= 0 && row + dr < this.size) && (col + dc >= 0 && col + dc < this.size)) {

                                // This neighbor is safe to add!
                                //System.out.println(population[row + dr][col + dc]);
                                //System.out.println("Row:" + row + dr + "Col" + col + dc);
                                this.population[row][col].addNeighbor(this.population[row + dr][col + dc]);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Apply the state changes of the SIR cells to progress the simulation by 1 time step.
     */
    public void update( ){

        // Ask each cell to predict its own next state
        for (int row = 0; row < this.size; row++) {
            for (int col = 0; col < this.size; col++) {
                this.population[row][col].updateNextState();
            }
        }

        // Ask each cell to update its own state
        for (int row = 0; row < this.size; row++) {
            for (int col = 0; col < this.size; col++) {
                this.population[row][col].updateCurrentState();
            }
        }
    }

    /**
     * Retrieve a copy of the population's current state fields, using a boolean 2D array to protect the cells
     * themselves from unwanted manipulation.
     * @return snapshot boolean[][], a map of current states
     */
    public SIRState[][] getPopulation(){
        SIRState[][] snapshot = new SIRState[this.size][this.size];
        for (int row=0; row<this.size; row++){
            for (int col=0; col<this.size; col++){
                snapshot[row][col] = this.population[row][col].getCurrentState();
            }
        }
        return snapshot;
    }
    /**
     * Retrieve a copy of the population's cell type, using a character 2D array to protect the cells
     * themselves from unwanted manipulation.
     * @return snapshot char[][], a map of cell types
     */
    public char[][] getPopulationTypes(){
        char[][] snapshot = new char[this.size][this.size];
        for (int row=0; row<this.size; row++){
            for (int col=0; col<this.size; col++){
                snapshot[row][col] = this.population[row][col].getType();
            }
        }
        return snapshot;
    }
}