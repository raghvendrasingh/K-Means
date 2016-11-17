/**
  * Created by rasingh on 11/3/16.
  */

import breeze.linalg._

import scala.collection.mutable._
import scala.util.Random
import scala.util.control._
import java.io.{BufferedWriter, File, FileWriter}


class KMeans(numCentres: Int, maxIter: Int = 2000, eps: Double = 0.01) {
  private val indexVectorMap = HashMap[Int, DenseVector[Double]]()
  private val labelVectorIndicesMap = HashMap[Int, ListBuffer[Int]]()
  private var vectorDimension = 0
  private var K = numCentres

  lazy val vectorLabelList = {
    val res = ListBuffer[(Vector[Double], Int)]()
    for (key <- indexVectorMap.keys) {
      val label = getLabel(key)
      assert(label != -1)
      res += ((indexVectorMap(key).toVector, label))
    }
    res.toList
  }

  lazy val label = {
    val res = ListBuffer[Int]()
    for (vectorLabel <- vectorLabelList) res += vectorLabel._2
    res.toList
  }


  private def populateIndexVectorMap(data: Vector[Vector[Double]]): Unit = {
    for (i <- 0 until data.length) indexVectorMap(i) = data(i).toDenseVector
  }

  private def getKDistinctRandomIndices(k: Int, maxVal: Int): Set[Int] = {
    val ind = Set[Int]()
    while (ind.size < k) {
      ind += Random.nextInt(maxVal)
    }
    assert(ind.size == k)
    ind
  }

  private def getKRandomCenters(data: Vector[Vector[Double]]): List[DenseVector[Double]] = {
    val randIndices = getKDistinctRandomIndices(K, data.length)
    randIndices.map(data(_).toDenseVector).toList
  }

  private def getIndexUsingWeightedProbabilityDistribution(centerIndices: ListBuffer[Int]): Int = {
    val centers = centerIndices.map(index => indexVectorMap(index)).toList
    // can be made more efficient by not calculating squaredDistances
    val squaredDistances = (indexVectorMap -- centerIndices).map{ case (ind,vector) => (ind,getNearestCenterIndexAndDistance(centers, ind)._2)}.toList
    val total = squaredDistances.foldLeft(0.0)((a,b) => a + b._2)
    val rand = Random.nextDouble()
    var cumulativeSum  = squaredDistances(0)._2/total
    var i = 0
    while (i < squaredDistances.length && cumulativeSum < rand) {
      i += 1
      cumulativeSum += squaredDistances(i)._2/total
    }
    squaredDistances(i)._1
  }

  private def getCentersUsingGonzalez(): List[DenseVector[Double]] = {
    val firstCenterIndex = Random.nextInt(indexVectorMap.size)
    val centerIndices = ListBuffer[Int](firstCenterIndex)
    while(centerIndices.length < K) {
      centerIndices += getIndexUsingWeightedProbabilityDistribution(centerIndices)
    }
    centerIndices.map(index => indexVectorMap(index)).toList
  }

  private def getInitialCentres(data: Vector[Vector[Double]], method: String): List[DenseVector[Double]] = {
    method match {
      case "random" => getKRandomCenters(data)
      case "gonzalez" => getCentersUsingGonzalez()
    }
  }

  private def distance(x1: DenseVector[Double], x2: DenseVector[Double]): Double = math.sqrt( sum( (x1 - x2) :^2.0 ) )

  private def getNearestCenterIndexAndDistance(centres: List[DenseVector[Double]], vectorIndex: Int): (Int,Double) = {
    val dataPoint = indexVectorMap(vectorIndex)
    var minDist = Double.MaxValue
    var minInd = -1
    for (i <- centres.indices) {
      val dist = distance(dataPoint, centres(i))
      if ( dist < minDist) {
        minDist = dist
        minInd = i
      }
    }
    (minInd, minDist*minDist)
  }

  private def initializeLabelVectorIndicesMap(numCentres: Int): Unit = {
    for (i <- 0 until numCentres) {
      labelVectorIndicesMap(i) = ListBuffer[Int]()
    }
  }

  private def updateLabelVectorIndexMap(centres: List[DenseVector[Double]]): Double = {
    var rss = 0.0
    initializeLabelVectorIndicesMap(K)
    for ( key <- indexVectorMap.keys) {
      val tup = getNearestCenterIndexAndDistance(centres, key)
      if (labelVectorIndicesMap.contains(tup._1)) labelVectorIndicesMap(tup._1) += key
      else labelVectorIndicesMap(tup._1) = ListBuffer(key)
      rss += tup._2
    }
    rss
  }

  private def getRandomlyInitializedVector(dimension: Int): DenseVector[Double] = {
    val res = DenseVector.fill(dimension, 0.0)
    for (i <- 0 until dimension) {
      res(i) = Random.nextDouble()
    }
    res
  }

  private def getAverageVector(vectorIndices: ListBuffer[Int]): DenseVector[Double] = {
    var avg = DenseVector.fill(vectorDimension,0.0)
    if (vectorIndices.isEmpty) getRandomlyInitializedVector(vectorDimension)
    else {
      for (ind <- vectorIndices) {
        avg += indexVectorMap(ind)
      }
      avg = avg :/ vectorIndices.length.toDouble
      avg
    }
  }

  private def getNewCentres(): List[DenseVector[Double]] = {
    var centres = List[DenseVector[Double]]()
    for (label <- labelVectorIndicesMap.keys) {
      centres = centres :+ getAverageVector(labelVectorIndicesMap(label))
    }
    centres
  }

  private def fitTransformUtil(centres: List[DenseVector[Double]], rss: Double): Unit = {
    var centresCurrent = centres
    var rssCurrent = rss
    var rssPrev = rssCurrent
    var iter = 0
    do {
      iter = iter + 1
      println(s"Iteration $iter")
      rssPrev = rssCurrent
      centresCurrent = getNewCentres()
      rssCurrent = updateLabelVectorIndexMap(centresCurrent)
    } while(math.abs(rssCurrent - rssPrev) > eps && iter < maxIter)
  }

  def fitTransform(data: Vector[Vector[Double]], initialCentersChoosingMethod: String = "gonzalez"): Unit = {
    vectorDimension = data(0).length
    K = numCentres
    populateIndexVectorMap(data)
    val centresPrev =  getInitialCentres(data, initialCentersChoosingMethod)
    val centresPrevTemp = List(DenseVector(5.8,6.2),DenseVector(10.2,9.8),DenseVector(14.2,13.8))
    val rssPrev = updateLabelVectorIndexMap(centresPrev)
    fitTransformUtil(centresPrev, rssPrev)
  }

  private def getLabel(ind: Int): Int = {
    var label = -1
    val loop = new Breaks
    loop.breakable {
      for (key <- labelVectorIndicesMap.keys) {
        if (labelVectorIndicesMap(key).contains(ind)) {
          label = key
          loop.break
        }
      }
    }
    label
  }


  def writeToCSV(fileName: String, cluster: DenseMatrix[Double]): Unit = {
    val csvFile =  new File(fileName)
    csvwrite(csvFile, cluster, separator = ',')
  }

  def denseMatrixToVectorOfVectors(mat: DenseMatrix[Double]): Vector[Vector[Double]] = {
    val result = Vector.fill[Vector[Double]](mat.rows)(Vector.fill[Double](mat.cols)(0.0))
    for (i <- 0 until mat.rows) {
      for (j <- 0 until mat.cols) {
        result(i)(j) = mat(i,j)
      }
    }
    result
  }

  def appendClusters(clusters: List[Vector[Vector[Double]]]): Vector[Vector[Double]] = {
    val totalVecs = clusters.foldLeft(0)((a,b) => a + b.length)
    val vecDimension = clusters(0)(0).length
    val result = Vector.fill[Vector[Double]](totalVecs)(Vector.fill[Double](vecDimension)(0.0))
    var index = 0
    for (k <- clusters.indices) {
      for (i <- 0 until clusters(k).length) {
        for (j <- 0 until clusters(k)(i).length) result(index)(j) = clusters(k)(i)(j)
        index += 1
      }
    }
    result
  }
}


object ExecuteKMeans extends App {
  val obj = new KMeans(3)
  val obj1  = new RandomGaussianSamples
  val alp1 = obj1.getRandomGaussianSamples(50, 2, 4.0, 1.0)
  val alp2 = obj1.getRandomGaussianSamples(50, 2, 8.0, 1.0)
  val alp3 = obj1.getRandomGaussianSamples(50, 2, 10.0, 1.0)
  obj.writeToCSV("cluster1.csv", alp1)
  obj.writeToCSV("cluster2.csv", alp2)
  obj.writeToCSV("cluster3.csv", alp3)
  val cluster1 = obj.denseMatrixToVectorOfVectors(alp1)
  val cluster2 = obj.denseMatrixToVectorOfVectors(alp2)
  val cluster3 = obj.denseMatrixToVectorOfVectors(alp3)
  val data = obj.appendClusters(List(cluster1,cluster2,cluster3))
  obj.fitTransform(data, initialCentersChoosingMethod = "gonzalez")
  val vectorLabelList = obj.vectorLabelList

  val file = new File("vector_and_labels.csv")
  val bw = new BufferedWriter(new FileWriter(file))
  bw.write("f1,f2,label" + "\n")
  for (elem <- vectorLabelList) {
    var str = ""
    for (dim <- elem._1) str = str + dim + ","
    str = str + elem._2
    bw.write(str + "\n")
  }
  bw.close()
}
