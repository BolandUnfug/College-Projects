/**
 *  Defines the cell type MaskedSIRCell, which represents a cell wearing a mask. <br>
 *  @author Boland Unfug, Caitrin Eaton, spring 2021
 */

public class QuarintineSIRCell extends SIRCell {
    public QuarintineSIRCell( SIRState initState ) {
        this.currentState = initState;
        this.nextState = this.currentState;
        this.totalNeighbors = 0;
        this.infectedNeighbors = 0;
        this.celltype = 'q';
        this.neighborhood = new SIRCell[ NEIGHBORHOOD_SIZE ]; // references are initially null
    }

    public void updateNextState(){
        if(this.currentState == SIRState.SUSCEPTIBLE){
            for(int i = 0; i<this.totalNeighbors; i++ ){
                if (this.neighborhood[i].getCurrentState() == SIRState.INFECTIOUS ) {
                    if(rand.nextDouble() < neighborhood[i].getInfectionRate()){
                        this.nextState = SIRState.INFECTIOUS;
                        this.infectionRate = 0;
                        
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
}

