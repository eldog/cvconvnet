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
 * \brief Declaration of convolutional network class
 *
 * This is the header file declaring CvConvNet class.
 * You need to include this header file in order to use 
 * this convolutional network library
 * \author Akhmed Umyarov
 * \date 2007
 */

#ifndef CVCONVNET_H
#define CVCONVNET_H

#include <opencv/cv.h>
#include <string>
#include <vector>
#include <map>

class CvGenericPlane;


//! The class represents the convolutional neural network
/*! The class is a container of individual feature maps (called planes)
of the convolutional neural network. The class also defines basic methods 
to operate with the network such as saving the network
as an XML string, loading the network from a string, 
accessing values of individual planes inside hidden 
layers etc. */
class CvConvNet
{
public:
		//! Constructor
		CvConvNet ( );

		//! Destructor
		virtual ~CvConvNet ( );

		//! Forward-propagation of input image through the whole network
		double fprop (CvArr *input);

		//! Provides access to individual planes inside the network
		const CvMat * getplane( std::string id );

		//! Produces string representation of the convolutional net
		std::string toString();

		//! Creates the convolutional net from a string representation
		int fromString ( std::string xml );

		//! Output of the network into stream
		friend std::ostream& operator<< (std::ostream& s, CvConvNet& n);

		//! [Not implemented yet]
		friend std::istream& operator>> (std::istream& s, CvConvNet& n);

protected:
		//! The container of the planes
		std::vector<CvGenericPlane *> m_plane;

		//! Hash table mapping string ids into int ids
		std::map<std::string, int> m_idmap; 

		std::string m_creator; //!< Name of creator of the network
		std::string m_name; //!< Name of the network itself
		std::string m_info; //!< Any additional info about the network
};

/*!
\example testimg.cpp
File testimg.cpp provides a simple example of how to use CvConvNet object
\example sample.xml
File sample.xml provides a sample convolutional network configuration.
\example testmnist.cpp
File testmnist.cpp provides a little more complicated example of using the class.
The program runs over the entire MNIST test dataset and calculates error rate.
*/

#endif // CVCONVNET_H
