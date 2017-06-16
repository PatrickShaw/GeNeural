namespace GeNeural.Training {
    public interface ISupervisedTrainer<T> {
        /// <summary>
        /// Trains a given trainable entity to output a certain set of outputs, given a certain set of inputs.
        /// </summary>
        /// <param name="trainable">The object being trained.</param>
        /// <param name="trainingInputs">The inputs that the trainable will use to train.</param>
        /// <param name="trainingOutputs">The outputs that the trainable needs to reproduce given the training inputs.</param>
        void Train(T trainable, double[] trainingInputs, double[] trainingOutputs);
    }
}
