namespace GeNeural.Training {
    public interface ISupervisedTrainer<T> {
        /// <summary>
        /// Trains a given trainable entity to output a certain set of outputs, given a certain set of inputs.
        /// </summary>
        /// <param name="trainable"></param>
        /// <param name="trainingInputs"></param>
        /// <param name="trainingOutputs"></param>
        void Train(T trainable, double[] trainingInputs, double[] trainingOutputs);
    }
}
