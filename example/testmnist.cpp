/*****************************************************************************
 IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING. By
downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install, copy or
use the software.

Contributors License Agreement

Copyright© 2007, Akhmed Umyarov. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.
- The name of Contributor may not be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
All information provided related to future Intel products and plans is
preliminary and subject to change at any time, without notice.
*****************************************************************************/

/*!\file
 * \brief Sample program using CvConvNet object
 * \author Akhmed Umyarov
 * \date 2007
 */

#include "cvconvnet.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>

using namespace std;

/*!
 * The function that runs the network over
 * the entire MNIST test dataset.
 * \return Exit code
 */
int main(int argc, char *argv[])
{
	if (argc <=1 )
	{
		cerr << "Usage: " << endl << "\ttestmnist <network.xml>" << endl;
		return 1;
	}

	// Create empty net object
	CvConvNet net;

	// Load mnist.xml file into string
	ifstream ifs( argv[1] );
	string xml ( (istreambuf_iterator<char> (ifs)) , istreambuf_iterator<char>() );
	
	// Create network from XML string
	if ( !net.fromString(xml) )
	{
		cerr << "*** ERROR: Can't load net from XML string" << endl;
		return 1;
	}

	// Represent MNIST datafiles as C++ file streams f1 and f2 respectively
	ifstream f1("t10k-images-idx3-ubyte",ios::in | ios::binary); // image data
	ifstream f2("t10k-labels-idx1-ubyte",ios::in | ios::binary); // label data
	
	if (!f1.is_open() || !f2.is_open())
	{
		cerr << "ERROR: Can't open MNIST files. Please locate them in current directory" << endl;
		return 1;
	}
	// Create buffers for image data and correct labels
	const int BUF_SIZE = 2048;
	char *buffer = new char[BUF_SIZE];
	char *label = new char[2];

	// Block for catching file exceptions
	try
	{
		// Read headers
		f1.read(buffer,16);
		f2.read(buffer,8);
	
		// Here is our info
		int imgno = 10000; // 10'000 images in file
		int imgheight = 28; // image size
		int imgwidth = 28;
		int imgpadx = 2; // Pad images by 2 black pixels, so
		int imgpady = 2; // the image becomes 32x32
		int imgpaddedheight = imgheight+2*imgpady; // padded image size
		int imgpaddedwidth = imgwidth+2*imgpadx;
		
		// Prepare image structures
		IplImage *img32 = cvCreateImageHeader( cvSize(imgpaddedheight,imgpaddedwidth), IPL_DEPTH_8U, 1 );
	
		// imageData now points to our buffer
		img32->imageData = buffer;
	
		// Clean the buffer
		memset(buffer,0,BUF_SIZE);
	
		// Initialize error counter
		int errors = 0;
	
		// Now cycle over all images in MNIST test dataset
		for (int i=0; i<imgno; i++)
		{
			// Load the image from file stream into img32
			// (remember img32->imgData points to our buffer)
			for (int k=0; k<imgheight; k++)
			{
				// Image in file is stored as 28x28, so we need to pad it to 32x32
				// So we read the image row-by-row with proper padding adjustments
				f1.read(&buffer[imgpadx+(imgpaddedwidth)*(k+2)],imgwidth);
			}
	
			// Propagate the matrix through network and get the result
			int pos = (int) net.fprop(img32);
			
			// Now read the correct label from label file stream
			f2.read(label,1);
	
			// Check if our prediction is correct
			if ( label[0]!=pos ) errors++;
		}
		
		// Print the error rate
		cout << "Error rate: " << (double)100.0*errors/imgno << "%" << endl;
		
	} catch (exception &e)
	{
		cerr << "Exception: " << e.what() << endl;
	}


	// Don't forget to free the memory
	delete[] label;
	delete[] buffer;

	// That's it!
	return 0;
}
