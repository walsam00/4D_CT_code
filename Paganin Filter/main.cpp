/*******************************************************
* Copyright (c) 2019, Maxim Puchkov
* ArryFire version of Paganin Filter
* Paganin et al., J.Microscopy, 2002
* Based on Python code from Diamond Source (https://github.com/DiamondLightSource/Savu)
*
********************************************************/
//#include <cmath>
//#include <cstdio>
//#include <cstdlib>
#include <arrayfire.h>
#include "cxxopts.hpp"

#include <H5Cpp.h>

using namespace af;


//ArryFire does not have FFTshift function but it has shift function
#define fftshift(in)  af::shift(in, in.dims(0)/2, in.dims(1)/2)

array paganin(const array &input, float aDistance, float aEnergy, float aResolution, float beta, float delta)
{
	array gray = input.as(f32); //colorSpace(input, AF_GRAY, AF_GRAY);

	//there are strange dimension limitations for ArrayFire FFT2D, use either exponents of 2 or prime factor <13

    int w = gray.dims(1)/1;
    int h = gray.dims(0)/1;

    printf("Image (%dx%d)\n", w,h);

    float micron = 1e-6; //10**(-6)
    float keV = 1000.0f;
    float distance = aDistance/1000.0f; //meters
    float energy = aEnergy * keV;
    float resolution = aResolution * micron;
    float wavelength = (1240.0 / energy) * 1e-9;
    float ratio = delta/beta;

    printf("Parameters: Distance: %f, Energy: %f, Resolution: %f, Wavelength: %f, Ratio: %f\n", distance, energy, resolution, wavelength, ratio);
    int padTopBottom = 0;
    int padLeftRight = 0;


    int height1 = h+2*padTopBottom;
    int width1 = w+2*padLeftRight;
    int centery = ceil(height1/2.0f)-1;
    int centerx = ceil(width1/2.0f)-1;


    double dpx =1.0/(width1*resolution);
    double dpy =1.0/(height1*resolution);

    array pxlist = (af::range(width1)-centerx)*dpx;
    array pylist = (af::range(height1)-centery)*dpy;

    array pxx = af::constant(0, height1, width1);
    array pyy = af::constant(0, height1, width1);
    pyy =af::tile(pylist, 1,width1);
    pxx =af::tile(af::reorder(pxlist, 1, 0), height1,1);

    array pd = (pxx*pxx+pyy*pyy)*wavelength*distance*af::Pi;

    array filterRe = 1.0f+ratio*pd;

    array filterComplex = af::complex(filterRe, filterRe);

    array forwardFourier = af::fft2(gray);

    array eq9 = fftshift(forwardFourier)/filterComplex;
    array eq13 = af::ifft2(eq9);

    return (-0.5f*ratio*af::log(af::abs(eq13)+0.00f)).as(f32);
}


int main(int argc, char **argv)
{
	cxxopts::Options options("Paganin GPU Filter", "Phase retrieval projections filtering on GPUs");


	bool useHDF = false;
	int device = 0;
    float distance = 12.0f; //meters
    float energy = 19.999;
    float resolution = 0.65f;
    float beta = 2e-9;
    float delta = 1e-7;
    std::string inputFileName;
    std::string outpuFileName;
    std::string datasetName = "normalized_stitched";

    options
      .allow_unrecognised_options()
      .add_options()
      ("e,energy", "Beam Energy(keV)", cxxopts::value<float>(energy))
      ("gpu", "GPU Device to use", cxxopts::value<int>(device))
      ("d,distance", "Distance from detector (mm)", cxxopts::value<float>(distance))
      ("r, resolution", "Pixel resolution(microns)", cxxopts::value<float>(resolution))
      ("delta", "Paganin Delta parameter", cxxopts::value<float>())
      ("beta", "Paganin Beta parameter", cxxopts::value<float>(beta))
      ("i, input", "Input File", cxxopts::value<std::string>(), "FILE")
      ("n, dataset", "Dataset Name of HDF5", cxxopts::value<std::string>()->default_value("normalized_stitched"))
      ("o,output", "Output file", cxxopts::value<std::string>()->default_value("paganin.tiff"))
      ("help", "Print help")

;

	auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
      std::cout << options.help({"", "Group"}) << std::endl;
      exit(0);
    }

    if (result.count("e"))
    {
    	energy = result["e"].as<float>();
    }

    if (result.count("d"))
    {
    	distance = result["d"].as<float>();
    }

    if (result.count("gpu"))
    {
    	device = result["gpu"].as<int>();
    }

    if (result.count("r"))
    {
    	resolution = result["r"].as<float>();
    }
    if (result.count("delta"))
    {
    	delta = result["delta"].as<float>();
    	std::cout<< "Delta " << delta << std::endl;
    }
    if (result.count("beta"))
    {
    	beta = result["beta"].as<float>();
    	std::cout<< "Beta " << beta << std::endl;
    }

    if (result.count("input"))
    {
    	std::string ifn(result["input"].as<std::string>());
      std::cout << "Input = " << result["input"].as<std::string>()
        << std::endl;
      inputFileName = ifn;
      if(ifn.substr(ifn.find_last_of(".") + 1) == "h5") {
    	useHDF = true;
      }
    }

    if (result.count("output"))
    {
      std::cout << "Output = " << result["output"].as<std::string>()
        << std::endl;
      outpuFileName = result["output"].as<std::string>();
    }

    if (result.count("n"))
    {
      std::cout << "n = " << result["n"].as<std::string>()
        << std::endl;
      datasetName = result["n"].as<std::string>();
    }


    try {
        af::setDevice(device);
        af::info();
        //array imgANKA = loadImage("/home/admin/cuda-workspace/paganin/data/ANKAPhase.tif", false);

        H5::H5File input(inputFileName.c_str(), H5F_ACC_RDWR);
        H5::DataSet dataset = input.openDataSet(datasetName.c_str());

        H5T_class_t type_class = dataset.getTypeClass();


        H5::DataSpace dataspace = dataset.getSpace();
        /*
         * Get the number of dimensions in the dataspace.
         */
        int rank = dataspace.getSimpleExtentNdims();
        /*
         * Get the dimension size of each dimension in the dataspace and
         * display them.
         */
        hsize_t dims_out[3];
        int ndims = dataspace.getSimpleExtentDims( dims_out, NULL);
        std::cout << "rank " << rank << ", dimensions " <<
            (unsigned long)(dims_out[0]) << " x " <<
            (unsigned long)(dims_out[1]) << " x " << (unsigned long)(dims_out[2]) << std::endl;

        for (int i=0; i<dims_out[0]; ++i)
        {
			/*
			 * Define hyperslab in the dataset; implicitly giving strike and
			 * block NULL.
			 */
			hsize_t      offset[3];   // hyperslab offset in the file
			hsize_t      count[3];    // size of the hyperslab in the file
			offset[0] = i;
			offset[1] = 0;
			offset[2] = 0;
			count[0]  = 1;
			count[1]  = dims_out[1];
			count[2]  = dims_out[2];
			dataspace.selectHyperslab( H5S_SELECT_SET, count, offset );
			/*
			 * Define the memory dataspace.
			 */
			hsize_t     dimsm[3];              /* memory space dimensions */
			dimsm[0] = 1;
			dimsm[1] = dims_out[1];
			dimsm[2] = dims_out[2];
			H5::DataSpace memspace( rank, dimsm );
			/*
			 * Define memory hyperslab.
			 */
			hsize_t      offset_out[3];   // hyperslab offset in memory
			hsize_t      count_out[3];    // size of the hyperslab in memory
			offset_out[0] = 0;
			offset_out[1] = 0;
			offset_out[2] = 0;
			count_out[0]  = 1;
			count_out[1]  = dims_out[1];
			count_out[2]  = dims_out[2];
			memspace.selectHyperslab( H5S_SELECT_SET, count_out, offset_out );


			hsize_t dataLength = 1;
			for(int i = 1; i < rank; i++){
				dataLength *= dims_out[i];
			}

			array img;


			std::cout <<  "data length: " << dataLength << std::endl;
			float *temp = new float[dataLength];
			dataset.read( temp, H5::PredType::NATIVE_FLOAT, memspace, dataspace );
			img = array(dims_out[2], dims_out[1], temp).T();


//			af::Window wnd("PaganinGPU Filter Experiment");
//			printf("Press ESC while the window is in focus to exit\n");
//			while (!wnd.close()) {
//				wnd.grid(1, 1);
//
//				wnd(0, 0).image(img, "Original");
//				//wnd(0, 1).image(imgANKA, "ANKA Phase");
//				//wnd(0, 1).image(paganin_ar, "Paganin effect");
//				wnd.show();
//			}

			delete [] temp;

			//
	//        H5::DataSpace tempSpace(1, &dataLength);
	//
	//        // Read data to memory
	//        dataset.read(temp, H5::PredType::NATIVE_FLOAT, tempSpace, dataspace);

			//array img = loadImage(inputFileName.c_str(), false);

			array paganin_ar = paganin(img, distance, energy, resolution, beta, delta);

			float* hostMem;

			hostMem = paganin_ar.T().host<float>();

			dataset.write(hostMem, H5::PredType::NATIVE_FLOAT, memspace, dataspace);
			af::freeHost(hostMem);
			memspace.close();
			printf("Image processed: %d \n", i);
        }
        dataspace.close();
        input.close();
        //af::Window wnd("PaganinGPU Filter Experiment");
        //printf("Press ESC while the window is in focus to exit\n");
//        while (!wnd.close()) {
//            wnd.grid(1, 2);
//
//            wnd(0, 0).image(img, "Original");
//            //wnd(0, 1).image(imgANKA, "ANKA Phase");
//            wnd(0, 1).image(paganin_ar, "Paganin effect");
//            wnd.show();
//        }
        //saveImageNative(outpuFileName.c_str(), paganin_ar);
    }
    catch (af::exception& e) {

        fprintf(stderr, "%s\n", e.what());
        throw;
    }
    return 0;
}
