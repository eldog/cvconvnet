/***********************************************************************
 *
 *    Copyright (C) 2002 Leon Bottou, Yann Le Cun, AT&T Corp, NECI.
 *  Includes parts of TL3:
 *    Copyright (C) 1987-1999 Leon Bottou and Neuristique.
 *  Includes selected parts of SN3.2:
 *    Copyright (C) 1991-2001 AT&T Corp.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA
 *
 ***********************************************************************/
/*!\file
 * \brief Fast sigmoid approximation trick
 * \author Yann LeCun
 * \date 2007
 */

#include "cvfastsigmoid.h"

const double PR  =0.66666666;
const double PO  =1.71593428;
const double A0  =1.0;
const double A1  =0.125*PR;
const double A2  =0.0078125*PR*PR;
const double A3  =0.000325520833333*PR*PR*PR;

//! Function for fast approximation of sigmoid 
double DQstdsigmoid(double x)
{
	double y;

	if (x >= 0.0)
		if (x < 13.0)
			y = A0+x*(A1+x*(A2+x*(A3)));
		else
			return PO;
	else
		if (x > -13.0)
			y = A0-x*(A1-x*(A2-x*(A3)));
		else
			return -PO;

	y *= y;
	y *= y;
	y *= y;
	y *= y;

	return (x > 0.0) ? PO*(y-1.0)/(y+1.0) : PO*(1.0-y)/(y+1.0);
}
