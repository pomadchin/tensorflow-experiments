package demo

import org.tensorflow.DataType
import org.tensorflow.Graph
import org.tensorflow.Output
import org.tensorflow.Session
import org.tensorflow.Tensor
import org.tensorflow.TensorFlow

import spire.syntax.cfor._

import java.io.PrintStream
import java.nio.charset.Charset
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

import scala.collection.JavaConverters._


/** Sample use of the TensorFlow Java API to label images using a pre-trained model. */
// https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image
object LabelImage {
  private def printUsage(s: PrintStream) = {
    val url = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
    s.println("Java program that uses a pre-trained Inception model (http://arxiv.org/abs/1512.00567)")
    s.println("to label JPEG images.")
    s.println("TensorFlow version: " + TensorFlow.version)
    s.println()
    s.println("Usage: label_image <model dir> <image file>")
    s.println()
    s.println("Where:")
    s.println("<model dir> is a directory containing the unzipped contents of the inception model")
    s.println("            (from " + url + ")")
    s.println("<image file> is the path to a JPEG image file")
  }

  def main(args: Array[String]): Unit = {
    val modelDir = "/Users/daunnc/subversions/git/github/pomadchin/tensorflow-experiments/model/"
    val imageFile = "/Users/daunnc/subversions/git/github/pomadchin/tensorflow-experiments/data/grace_hopper.jpg"

    val graphDef = readAllBytesOrExit(Paths.get(modelDir, "tensorflow_inception_graph.pb"))
    val labels = readAllLinesOrExit(Paths.get(modelDir, "imagenet_comp_graph_label_strings.txt"))
    val imageBytes = readAllBytesOrExit(Paths.get(imageFile))

    val image = constructAndExecuteGraphToNormalizeImage(imageBytes)
    try {
      val labelProbabilities = executeInceptionGraph(graphDef, image)
      val bestLabelIdx = maxIndex(labelProbabilities)
      println(f"BEST MATCH: ${labels(bestLabelIdx)} (${labelProbabilities(bestLabelIdx) * 100f}%.2f likely)")
    } finally if (image != null) image.close()
  }

  private def constructAndExecuteGraphToNormalizeImage(imageBytes: Array[Byte]) = {
    val g = new Graph
    try {
      val b = new LabelImage.GraphBuilder(g)
      // Some constants specific to the pre-trained model at:
      // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
      //
      // - The model was trained with images scaled to 224x224 pixels.
      // - The colors, represented as R, G, B in 1-byte each were converted to
      //   float using (value - Mean)/Scale.
      val H = 224
      val W = 224
      val mean = 117f
      val scale = 1f
      // Since the graph is being constructed once per execution here, we can use a constant for the
      // input image. If the graph were to be re-used for multiple input images, a placeholder would
      // have been more appropriate.
      val input = b.constant("input", imageBytes)
      val output = b.div(b.sub(b.resizeBilinear(b.expandDims(b.cast(b.decodeJpeg(input, 3), DataType.FLOAT), b.constant("make_batch", 0)), b.constant("size", Array[Int](H, W))), b.constant("mean", mean)), b.constant("scale", scale))
      val s = new Session(g)
      try { s.runner.fetch(output.op.name).run.get(0) } finally if (s != null) s.close()
    } finally if (g != null) g.close()
  }

  private def executeInceptionGraph(graphDef: Array[Byte], image: Tensor) = {
    val g = new Graph
    try {
      g.importGraphDef(graphDef)
      val s = new Session(g)
      val result = s.runner.feed("input", image).fetch("output").run.get(0)
      try {
        val rshape = result.shape
        if (result.numDimensions != 2 || rshape(0) != 1)
          throw new RuntimeException(
            s"Expected model to produce a [1 N] shaped tensor where N is the number of labels " +
              s"instead it produced one with shape ${rshape}"
          )

        val nlabels = rshape(1).toInt
        result.copyTo(Array.ofDim[Float](1, nlabels))(0)
      } finally {
        if (s != null) s.close()
        if (result != null) result.close()
      }
    } finally if (g != null) g.close()
  }

  private def maxIndex(probabilities: Array[Float]) = {
    var best = 0
    cfor(1)(_ < probabilities.length, _ + 1) { i =>
      if (probabilities(i) > probabilities(best)) best = i
    }
    best
  }

  def readAllBytesOrExit(path: Path): Array[Byte] = Files.readAllBytes(path)
  def readAllLinesOrExit(path: Path): Seq[String] =
    Files.readAllLines(path, Charset.forName("UTF-8")).asScala

  // In the fullness of time, equivalents of the methods of this class should be auto-generated from
  // the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
  // like Python, C++ and Go.
  class GraphBuilder (var g: Graph) {
    def div(x: Output, y: Output): Output = binaryOp("Div", x, y)
    def sub(x: Output, y: Output): Output = binaryOp("Sub", x, y)
    def resizeBilinear(images: Output, size: Output): Output = binaryOp("ResizeBilinear", images, size)
    def expandDims(input: Output, dim: Output): Output = binaryOp("ExpandDims", input, dim)
    def cast(value: Output, dtype: DataType): Output = g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build.output(0)
    def decodeJpeg(contents: Output, channels: Long): Output = g.opBuilder("DecodeJpeg", "DecodeJpeg").addInput(contents).setAttr("channels", channels).build.output(0)
    def constant(name: String, value: Any): Output = {
      val t = Tensor.create(value)
      try {
        g.opBuilder("Const", name).setAttr("dtype", t.dataType).setAttr("value", t).build.output(0)
      } finally if (t != null) t.close()
    }
    def binaryOp(`type`: String, in1: Output, in2: Output): Output =
      g.opBuilder(`type`, `type`).addInput(in1).addInput(in2).build.output(0)
  }
}
