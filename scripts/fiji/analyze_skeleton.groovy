// Run through Fiji --headless --console --run. Python supplies the JSON path
// via the environment so spaces and punctuation in file paths remain safe.
import groovy.json.JsonSlurper
import groovy.json.JsonOutput
import ij.IJ
import ij.ImagePlus
import ij.io.FileSaver
import sc.fiji.analyzeSkeleton.AnalyzeSkeleton_

def configPath = System.getenv('PARSE_ANALYZE_CONFIG')
if (!configPath) {
    throw new IllegalArgumentException('PARSE_ANALYZE_CONFIG is required')
}
def config = new JsonSlurper().parse(new File(configPath))
def image = IJ.openImage(config.input_tiff as String)
if (image == null) {
    throw new IllegalArgumentException('Could not open the skeleton TIFF')
}
try {
    def calibration = image.getCalibration()
    calibration.pixelWidth = config.spacing_mm[0] as double
    calibration.pixelHeight = config.spacing_mm[1] as double
    calibration.pixelDepth = config.spacing_mm[2] as double
    calibration.setUnit('mm')
    image.setCalibration(calibration)

    def analyzer = new AnalyzeSkeleton_()
    analyzer.setup('', image)
    // This overload disables pruning, shortest paths, and GUI windows.
    def result = analyzer.run()
    def graphs = result.getGraph()
    def branchCounts = result.getBranches()
    def endpoints = result.getEndPoints()
    def junctions = result.getJunctions()
    def junctionVoxels = result.getJunctionVoxels()
    def graphComponents = []
    def coordinates = { points -> points.collect { point -> [point.x, point.y, point.z] } }

    new File(config.branches_csv as String).withWriter('UTF-8') { branches ->
        branches.write('component_id,branch_id,length_mm\n')
        new File(config.components_csv as String).withWriter('UTF-8') { components ->
            components.write('component_id,branch_count,endpoint_count,junction_count,junction_voxel_count,mean_branch_length_mm,max_branch_length_mm\n')
            for (int index = 0; index < result.getNumOfTrees(); index++) {
                def edges = graphs[index].getEdges()
                def vertices = graphs[index].getVertices()
                def nodeIds = new IdentityHashMap()
                vertices.eachWithIndex { vertex, nodeIndex -> nodeIds.put(vertex, nodeIndex + 1) }
                def nodes = vertices.collect { vertex ->
                    [node_id: nodeIds.get(vertex), voxels: coordinates(vertex.getPoints())]
                }
                def graphBranches = []
                if (edges.size() != branchCounts[index]) {
                    throw new IllegalStateException('Graph edges and reported branch count disagree')
                }
                double total = 0.0
                double maximum = 0.0
                for (int edgeIndex = 0; edgeIndex < edges.size(); edgeIndex++) {
                    def edge = edges[edgeIndex]
                    double length = edge.getLength()
                    if (!Double.isFinite(length) || length < 0.0) {
                        throw new IllegalStateException('Invalid branch length')
                    }
                    total += length
                    maximum = Math.max(maximum, length)
                    branches.write("${index + 1},${edgeIndex + 1},${length}\n")
                    if (!nodeIds.containsKey(edge.getV1()) || !nodeIds.containsKey(edge.getV2())) {
                        throw new IllegalStateException('Branch refers to a node absent from its graph')
                    }
                    graphBranches.add([
                        branch_id: edgeIndex + 1,
                        v1: nodeIds.get(edge.getV1()), v2: nodeIds.get(edge.getV2()),
                        slabs: coordinates(edge.getSlabs()), length_mm: length,
                    ])
                }
                double mean = edges.isEmpty() ? 0.0 : total / edges.size()
                components.write("${index + 1},${edges.size()},${endpoints[index]},${junctions[index]},${junctionVoxels[index]},${mean},${maximum}\n")
                graphComponents.add([component_id: index + 1, nodes: nodes, branches: graphBranches])
            }
        }
    }
    def saveStack = { stack, destination ->
        def output = new ImagePlus('Skeleton labels', stack)
        try {
            output.setCalibration(calibration.copy())
            def saver = new FileSaver(output)
            boolean saved = stack.getSize() == 1 ? saver.saveAsTiff(destination as String) :
                saver.saveAsTiffStack(destination as String)
            if (!saved) {
                throw new IOException('Could not save skeleton label stack')
            }
        } finally {
            output.close()
        }
    }
    saveStack(analyzer.getResultImage(false), config.voxel_types_tiff)
    saveStack(analyzer.getLabeledSkeletons(), config.components_tiff)
    new File(config.graph_json as String).setText(JsonOutput.toJson([
        schema_version: 2,
        coordinate_system: [space: 'voxel', axes: 'XYZ', index_base: 0,
                            reference_image: 'lung_arteries_skeleton.nii.gz'],
        shape: config.shape, affine: config.affine,
        components: graphComponents,
    ]), 'UTF-8')
    // Fiji may exit with status 0 even if a script throws. Only acknowledge a
    // completed analysis after every table, image, and graph has been saved.
    new File(config.completion_json as String).setText(JsonOutput.toJson([
        status: 'ok', component_count: result.getNumOfTrees(),
        branch_count: branchCounts.sum() ?: 0,
    ]), 'UTF-8')
} finally {
    image.close()
}
