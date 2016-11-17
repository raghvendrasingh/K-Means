import breeze.linalg.DenseMatrix

/**
  * Created by rasingh on 11/10/16.
  */
class RandomGaussianSamples {
   def getRandomGaussianSamples(numSamples: Int, numFeatures: Int, mean: Double, sd: Double): DenseMatrix[Double] = {
     val norm = breeze.stats.distributions.Gaussian(mean, sd)
     val samples = norm.sample(numSamples*numFeatures)
     new DenseMatrix[Double](numSamples, numFeatures, samples.toArray)
   }

}