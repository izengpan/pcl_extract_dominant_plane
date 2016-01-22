#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/boost.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>

typedef pcl::PointXYZRGBA PointType;
typedef pcl::PointCloud<PointType> Cloud;
typedef Cloud::Ptr CloudPtr;
typedef Cloud::ConstPtr CloudConstPtr;

bool default_keep_organized = true;
bool default_visualize = false;

void printHelp(int, char**argv) {
    pcl::console::print_error ("Syntax is: %s input.pcd output.pcd <options>\n", argv[0]);
    pcl::console::print_info  ("  where options are:\n");
    pcl::console::print_info  ("                     -keep 0/1 = keep the points organized (1) or not (default: ");
    pcl::console::print_value ("%d", default_keep_organized); pcl::console::print_info (")\n");
}

// Get the indices of the points in the cloud that belong to the dominant plane
pcl::PointIndicesPtr extractPlaneIndices(Cloud::Ptr cloud, double plane_thold = 0.02) {
    // Object for storing the plane model coefficients.
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    // Create the segmentation object.
    pcl::SACSegmentation<PointType> segmentation;
    segmentation.setInputCloud(cloud);
    // Configure the object to look for a plane.
    segmentation.setModelType(pcl::SACMODEL_PLANE);
    // Use RANSAC method.
    segmentation.setMethodType(pcl::SAC_RANSAC);
    // Set the maximum allowed distance to the model.
    segmentation.setDistanceThreshold(plane_thold);
    // Enable model coefficient refinement (optional).
    segmentation.setOptimizeCoefficients(true);

    pcl::PointIndicesPtr inlierIndices = boost::make_shared<pcl::PointIndices>();
    segmentation.segment(*inlierIndices, *coefficients);

    return inlierIndices;
//    if (inlierIndices.indices.size() == 0)
//        std::cout << "Could not find any points that fitted the plane model." << std::endl;
//    else
//    {
//        std::cerr << "Model coefficients: " << coefficients->values[0] << " "
//        << coefficients->values[1] << " "
//        << coefficients->values[2] << " "
//        << coefficients->values[3] << std::endl;
//
//        // Copy all inliers of the model to another cloud.
//        pcl::copyPointCloud<pcl::PointXYZ>(*cloud, inlierIndices, *inlierPoints);
//    }
}

CloudPtr extractIndices(Cloud::Ptr cloud, pcl::PointIndicesPtr indices, bool keep_organised) {
    // Object for extracting points from a list of indices.
    pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(indices);
    extract.setKeepOrganized(keep_organised);
    // We will extract the points that are indexed (the ones that are in a plane).
    extract.setNegative(false);

    Cloud::Ptr extractedPlane = boost::make_shared<Cloud>();
    extract.filter(*extractedPlane);
    return extractedPlane;
}



// Euclidean Clustering
pcl::PointIndicesPtr euclideanClustering(Cloud::Ptr cloud, double cluster_tolerance = 0.005, int min_size = 100, int max_size = 307200) {
    // kd-tree object for searches.
    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>);
    kdtree->setInputCloud(cloud);

    // Euclidean clustering object.
    pcl::EuclideanClusterExtraction<PointType> clustering;
    // Set cluster tolerance to 1cm (small values may cause objects to be divided
    // in several clusters, whereas big values may join objects in a same cluster).
    clustering.setClusterTolerance(cluster_tolerance);
    // Set the minimum and maximum number of points that a cluster can have.
    clustering.setMinClusterSize(min_size);
    clustering.setMaxClusterSize(max_size);
    clustering.setSearchMethod(kdtree);
    clustering.setInputCloud(cloud);
    std::vector<pcl::PointIndices> clusters;
    clustering.extract(clusters);

    // Find largest cluster and return indices
    size_t largest_cluster_index = 0;
    size_t current_cluster_index = 0;
    size_t max_points = 0;
    for(const auto& cluster: clusters) {
        if( cluster.indices.size() > max_points) {
            max_points = cluster.indices.size();
            largest_cluster_index = current_cluster_index;
        }
        current_cluster_index++;
    }

    pcl::PointIndicesPtr indicesPtr = boost::make_shared<pcl::PointIndices>(clusters.at(largest_cluster_index));
    return indicesPtr;
//    // For every cluster...
//    int currentClusterNum = 1;
//    for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
//    {
//        // ...add all its points to a new cloud...
//        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
//        for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
//            cluster->points.push_back(cloud->points[*point]);
//        cluster->width = cluster->points.size();
//        cluster->height = 1;
//        cluster->is_dense = true;
//
//        // ...and save it to disk.
//        if (cluster->points.size() <= 0)
//            break;
//        std::cout << "Cluster " << currentClusterNum << " has " << cluster->points.size() << " points." << std::endl;
//        std::string fileName = "cluster" + boost::to_string(currentClusterNum) + ".pcd";
//        pcl::io::savePCDFileASCII(fileName, *cluster);
//
//        currentClusterNum++;
//    }
}

// Region Growing
pcl::PointIndicesPtr regionGrowing(Cloud::Ptr cloud,
                                   double smoothness_degrees = 7.0,
                                   double curvature_thold = 1.0,
                                   double normal_radius_search = 0.01,
                                   size_t num_neighbours=30,
                                   size_t min_size=100,
                                   size_t max_size=307200 ) {
    // kd-tree object for searches.
    pcl::search::KdTree<PointType>::Ptr kdTree(new pcl::search::KdTree<PointType>);
    kdTree->setInputCloud(cloud);

    // Estimate the normals.
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<PointType, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(cloud);
    normalEstimation.setRadiusSearch(normal_radius_search);
    normalEstimation.setSearchMethod(kdTree);
    normalEstimation.compute(*normals);

    // Region growing clustering object.
    pcl::RegionGrowing<PointType, pcl::Normal> clustering;
    clustering.setMinClusterSize((int)min_size);
    clustering.setMaxClusterSize((int)max_size);
    clustering.setSearchMethod(kdTree);
    clustering.setNumberOfNeighbours((int)num_neighbours);
    clustering.setInputCloud(cloud);
    clustering.setInputNormals(normals);
    // Set the angle in radians that will be the smoothness threshold
    // (the maximum allowable deviation of the normals).
    clustering.setSmoothnessThreshold((float)(smoothness_degrees / 180.0 * M_PI)); // degrees.
    // Set the curvature threshold. The disparity between curvatures will be
    // tested after the normal deviation check has passed.
    clustering.setCurvatureThreshold((float)curvature_thold);

    std::vector <pcl::PointIndices> clusters;
    clustering.extract(clusters);

    // Find largest cluster and return indices
    size_t largest_cluster_index = 0;
    size_t current_cluster_index = 0;
    size_t max_points = 0;
    for(const auto& cluster: clusters) {
        if( cluster.indices.size() > max_points) {
            max_points = cluster.indices.size();
            largest_cluster_index = current_cluster_index;
        }
        current_cluster_index++;
    }

    pcl::PointIndicesPtr indicesPtr = boost::make_shared<pcl::PointIndices>(clusters.at(largest_cluster_index));
    return indicesPtr;
}

// Region Growing Color

int main(int argc, char** argv) {
    pcl::console::print_info ("Find the dominant plane in a point cloud. For more information, use: %s -h\n", argv[0]);

    if (argc < 3) {
        printHelp (argc, argv);
        return (-1);
    }

    bool batch_mode = false;

    // Command line parsing
    bool keep_organized = default_keep_organized;
    bool visualize = default_visualize;

    pcl::console::parse_argument (argc, argv, "-keep", keep_organized);
    if(pcl::console::find_switch (argc, argv, "-vis"))
        visualize = true;
    // Parse the command line arguments for .pcd files
    std::vector<int> p_file_indices;
    p_file_indices = pcl::console::parse_file_extension_argument(argc, argv, ".pcd");
    if (p_file_indices.size() != 2)
    {
        pcl::console::print_error ("Need one input PCD file and one output PCD file to continue.\n");
        return (-1);
    }

    // Load the first file
    Cloud::Ptr cloud(new pcl::PointCloud<PointType>());
    pcl::console::print_highlight("Loading ");
    pcl::console::print_value("%s ", argv[p_file_indices[0]]);
    std::string inputPCD(argv[p_file_indices[0]]);
    std::string outputPCD(argv[p_file_indices[1]]);

    if(!pcl::io::loadPCDFile(inputPCD, *cloud) < 0) {
        pcl::console::print_error("Cloud not load pcd file: %s", inputPCD.c_str());
        return -1;
    }
//    if (!pcl::io::loadPCDFile(inputPCD, *cloud))
//        return (-1);


    pcl::PointIndicesPtr pointIndices = extractPlaneIndices(cloud);

    if(pointIndices->indices.size() == 0) {
        pcl::console::print_highlight("No plane to be found in input pcd.\n");
        return 0;
    }

    Cloud::Ptr extractedPlane = extractIndices(cloud, pointIndices, keep_organized);

    pcl::PointIndicesPtr largestClusterIndices = euclideanClustering(extractedPlane);

    Cloud::Ptr clusteredPlane = extractIndices(cloud, largestClusterIndices, keep_organized);

    pcl::PointIndicesPtr regionGrowIndices = regionGrowing(clusteredPlane);

    Cloud::Ptr finalPlane = extractIndices(cloud, regionGrowIndices, keep_organized);

    if(visualize) {
        pcl::visualization::CloudViewer viewer("Cloud Viewer");
        viewer.showCloud(finalPlane);
        while (!viewer.wasStopped())
        {
            // Do nothing but wait.
        }
    }

    pcl::io::savePCDFileBinaryCompressed(outputPCD, *finalPlane);

    return 0;



//    // Perform the feature estimation
//    Cloud::Ptr output (new Cloud);
//    compute (cloud, output, radius, inside, keep_organized);
//
//    // Save into the second file
//    saveCloud (argv[p_file_indices[1]], output);

}