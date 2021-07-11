/**
 *  Defines the cell type MaskedSIRCell, which represents a cell not wearing a mask. <br>
 *  @author Boland Unfug, Caitrin Eaton, spring 2021
 */

public class UnMaskedSIRCell extends SIRCell {
    public UnMaskedSIRCell( SIRState initState ) {
        this.currentState = initState;
        this.nextState = this.currentState;
        this.totalNeighbors = 0;
        this.infectedNeighbors = 0;
        this.celltype = 'u';
        this.neighborhood = new SIRCell[ NEIGHBORHOOD_SIZE ]; // references are initially null
    }
}

