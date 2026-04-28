// Copyright 2020 S. Hoseininasab

// This file is part of CADNA.

// CADNA is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// CADNA is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public License
// along with CADNA.  If not, see <http://www.gnu.org/licenses/>.

#ifndef CADNA_HALF_HPP
#define CADNA_HALF_HPP

#include "half.hpp"

using half_float::half;

inline bool operator<(half x, double y)
	{
		if ((double)x < y)
            return 1;
        else
            return 0;
	}
	inline bool operator<(half x, float y)
	{
		if ((double)x < y)
            return 1;
        else
            return 0;
	}
	inline bool operator<(half x, unsigned long long y)
	{
		if ((double)x < y)
            return 1;
        else
            return 0;
	}
   inline bool operator<(half x, long long y)
	{
		if ((double)x < y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<(half x, unsigned long y)
	{
		if ((double)x < y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<(half x, long y)
	{
		if ((double)x < y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<(half x, unsigned int y)
	{
		if ((double)x < y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<(half x, int y)
	{
		if ((double)x < y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<(half x, unsigned short y)
	{
		if ((double)x < y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<(half x, short y)
	{
		if ((double)x < y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<(half x, unsigned char y)
	{
		if ((double)x < y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<(half x, char y)
	{
		if ((double)x < y)
            return 1;
        else
            return 0;
	}
	
		inline bool operator<(double x, half y)
	{
		if (x < (double)y)
            return 1;
        else
            return 0;
	}
	inline bool operator<(float x, half y)
	{
		if (x < (double)y)
            return 1;
        else
            return 0;
	}
	inline bool operator<(unsigned long long x, half y)
	{
		if (x < (double)y)
            return 1;
        else
            return 0;
	}
   inline bool operator<(long long x, half y)
	{
		if (x < (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<(unsigned long x, half y)
	{
		if (x < (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<(long x, half y)
	{
		if (x < (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<(unsigned int x, half y)
	{
		if (x < (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<(int x, half y)
	{
		if (x < (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<(unsigned short x, half y)
	{
		if (x < (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<(short x, half y)
	{
		if (x < (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<(unsigned char x, half y)
	{
		if (x < (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<(char x, half y)
	{
		if (x < (double)y)
            return 1;
        else
            return 0;
	}



inline bool operator>(half x, double y)
	{
		if ((double)x > y)
            return 1;
        else
            return 0;
	}
	inline bool operator>(half x, float y)
	{
		if ((double)x > y)
            return 1;
        else
            return 0;
	}
	inline bool operator>(half x, unsigned long long y)
	{
		if ((double)x > y)
            return 1;
        else
            return 0;
	}
   inline bool operator>(half x, long long y)
	{
		if ((double)x > y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>(half x, unsigned long y)
	{
		if ((double)x > y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>(half x, long y)
	{
		if ((double)x > y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>(half x, unsigned int y)
	{
		if ((double)x > y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>(half x, int y)
	{
		if ((double)x > y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>(half x, unsigned short y)
	{
		if ((double)x > y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>(half x, short y)
	{
		if ((double)x > y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>(half x, unsigned char y)
	{
		if ((double)x > y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>(half x, char y)
	{
		if ((double)x > y)
            return 1;
        else
            return 0;
	}
	
		inline bool operator>(double x, half y)
	{
		if (x > (double)y)
            return 1;
        else
            return 0;
	}
	inline bool operator>(float x, half y)
	{
		if (x > (double)y)
            return 1;
        else
            return 0;
	}
	inline bool operator>(unsigned long long x, half y)
	{
		if (x > (double)y)
            return 1;
        else
            return 0;
	}
   inline bool operator>(long long x, half y)
	{
		if (x > (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>(unsigned long x, half y)
	{
		if (x > (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>(long x, half y)
	{
		if (x > (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>(unsigned int x, half y)
	{
		if (x > (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>(int x, half y)
	{
		if (x > (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>(unsigned short x, half y)
	{
		if (x > (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>(short x, half y)
	{
		if (x > (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>(unsigned char x, half y)
	{
		if (x > (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>(char x, half y)
	{
		if (x > (double)y)
            return 1;
        else
            return 0;
	}





inline bool operator<=(half x, double y)
	{
		if ((double)x <= y)
            return 1;
        else
            return 0;
	}
	inline bool operator<=(half x, float y)
	{
		if ((double)x <= y)
            return 1;
        else
            return 0;
	}
	inline bool operator<=(half x, unsigned long long y)
	{
		if ((double)x <= y)
            return 1;
        else
            return 0;
	}
   inline bool operator<=(half x, long long y)
	{
		if ((double)x <= y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<=(half x, unsigned long y)
	{
		if ((double)x <= y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<=(half x, long y)
	{
		if ((double)x <= y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<=(half x, unsigned int y)
	{
		if ((double)x <= y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<=(half x, int y)
	{
		if ((double)x <= y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<=(half x, unsigned short y)
	{
		if ((double)x <= y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<=(half x, short y)
	{
		if ((double)x <= y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<=(half x, unsigned char y)
	{
		if ((double)x <= y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<=(half x, char y)
	{
		if ((double)x <= y)
            return 1;
        else
            return 0;
	}
	
		inline bool operator<=(double x, half y)
	{
		if (x <= (double)y)
            return 1;
        else
            return 0;
	}
	inline bool operator<=(float x, half y)
	{
		if (x <= (double)y)
            return 1;
        else
            return 0;
	}
	inline bool operator<=(unsigned long long x, half y)
	{
		if (x <= (double)y)
            return 1;
        else
            return 0;
	}
   inline bool operator<=(long long x, half y)
	{
		if (x <= (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<=(unsigned long x, half y)
	{
		if (x <= (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<=(long x, half y)
	{
		if (x <= (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<=(unsigned int x, half y)
	{
		if (x <= (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<=(int x, half y)
	{
		if (x <= (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<=(unsigned short x, half y)
	{
		if (x <= (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<=(short x, half y)
	{
		if (x <= (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<=(unsigned char x, half y)
	{
		if (x <= (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator<=(char x, half y)
	{
		if (x <= (double)y)
            return 1;
        else
            return 0;
	}


inline bool operator>=(half x, double y)
	{
		if ((double)x >= y)
            return 1;
        else
            return 0;
	}
	inline bool operator>=(half x, float y)
	{
		if ((double)x >= y)
            return 1;
        else
            return 0;
	}
	inline bool operator>=(half x, unsigned long long y)
	{
		if ((double)x >= y)
            return 1;
        else
            return 0;
	}
   inline bool operator>=(half x, long long y)
	{
		if ((double)x >= y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>=(half x, unsigned long y)
	{
		if ((double)x >= y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>=(half x, long y)
	{
		if ((double)x >= y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>=(half x, unsigned int y)
	{
		if ((double)x >= y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>=(half x, int y)
	{
		if ((double)x >= y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>=(half x, unsigned short y)
	{
		if ((double)x >= y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>=(half x, short y)
	{
		if ((double)x >= y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>=(half x, unsigned char y)
	{
		if ((double)x >= y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>=(half x, char y)
	{
		if ((double)x >= y)
            return 1;
        else
            return 0;
	}
	
		inline bool operator>=(double x, half y)
	{
		if (x >= (double)y)
            return 1;
        else
            return 0;
	}
	inline bool operator>=(float x, half y)
	{
		if (x >= (double)y)
            return 1;
        else
            return 0;
	}
	inline bool operator>=(unsigned long long x, half y)
	{
		if (x >= (double)y)
            return 1;
        else
            return 0;
	}
   inline bool operator>=(long long x, half y)
	{
		if (x >= (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>=(unsigned long x, half y)
	{
		if (x >= (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>=(long x, half y)
	{
		if (x >= (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>=(unsigned int x, half y)
	{
		if (x >= (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>=(int x, half y)
	{
		if (x >= (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>=(unsigned short x, half y)
	{
		if (x >= (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>=(short x, half y)
	{
		if (x >= (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>=(unsigned char x, half y)
	{
		if (x >= (double)y)
            return 1;
        else
            return 0;
	}
	
	inline bool operator>=(char x, half y)
	{
		if (x >= (double)y)
            return 1;
        else
            return 0;
	}



    inline double operator+(double x, half y)
	{
    
       return x + (double)y;
        
    }
    
    inline float operator+(float x, half y)
	{
       return x + (float)y;
        
    }
    
    inline half operator+(unsigned long long x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x + (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator+(long long x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x + (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator+(unsigned long x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x + (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator+(long x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x + (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator+(unsigned int x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x + (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator+(int x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x + (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator+(unsigned short x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x + (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator+(short x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x + (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator+(unsigned char x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x + (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator+(char x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x + (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }




   inline double operator+( half x, double y)
	{
    
       return (double)x + y;
        
    }
    
    inline float operator+(half x, float y)
	{
       return (float)x + y;
        
    }
    
    inline half operator+( half x, unsigned long long y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x + y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator+(half x, long long y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x + y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator+(half x, unsigned long y)
	{
       double tmp;
       half tmp1;
       tmp =(double)x + y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator+(half x, long y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x + y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator+(half x, unsigned int y)
	{
       double tmp;
       half tmp1;
       tmp =(double)x + y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator+(half x, int y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x + y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator+(half x, unsigned short y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x + y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator+(half x, short y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x + y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator+( half x, unsigned char y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x + y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator+( half x, char y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x + y;
       tmp1 = tmp;
       return tmp1;
        
    }



    inline double operator-(double x, half y)
	{
       return x - (double)y;
        
    }
    
    inline float operator-(float x, half y)
	{
       return x - (float)y;
        
    }
    
    inline half operator-(unsigned long long x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x - (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator-(long long x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x - (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator-(unsigned long x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x - (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator-(long x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x - (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator-(unsigned int x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x - (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator-(int x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x - (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator-(unsigned short x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x - (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator-(short x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x - (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator-(unsigned char x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x - (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator-(char x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x - (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    

    inline double operator-( half x, double y)
	{
    
       return (double)x - y;
        
    }
    
    inline float operator-(half x, float y)
	{
       return (float)x - y;
        
    }
    
    inline half operator-( half x, unsigned long long y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x - y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator-(half x, long long y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x - y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator-(half x, unsigned long y)
	{
       double tmp;
       half tmp1;
       tmp =(double)x - y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator-(half x, long y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x - y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator-(half x, unsigned int y)
	{
       double tmp;
       half tmp1;
       tmp =(double)x - y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator-(half x, int y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x - y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator-(half x, unsigned short y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x - y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator-(half x, short y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x - y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator-( half x, unsigned char y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x - y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator-( half x, char y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x - y;
       tmp1 = tmp;
       return tmp1;
        
    }






    inline double operator*(double x, half y)
	{
       return x * (double)y;
        
    }

    inline float operator*(float x, half y)
	{
       return x * (float)y;
        
    }
    
    inline half operator*(unsigned long long x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x * (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator*(long long x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x * (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator*(unsigned long x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x * (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator*(long x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x * (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator*(unsigned int x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x * (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator*(int x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x * (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator*(unsigned short x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x * (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator*(short x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x * (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator*(unsigned char x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x * (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator*(char x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x * (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }



    inline double operator*( half x, double y)
	{
    
       return (double)x * y;
        
    }
    
    inline float operator*(half x, float y)
	{
       return (float)x * y;
        
    }
    
    inline half operator*( half x, unsigned long long y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x * y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator*(half x, long long y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x * y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator*(half x, unsigned long y)
	{
       double tmp;
       half tmp1;
       tmp =(double)x * y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator*(half x, long y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x * y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator*(half x, unsigned int y)
	{
       double tmp;
       half tmp1;
       tmp =(double)x * y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator*(half x, int y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x * y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator*(half x, unsigned short y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x * y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator*(half x, short y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x * y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator*( half x, unsigned char y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x * y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator*( half x, char y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x * y;
       tmp1 = tmp;
       return tmp1;
        
    }







    inline double operator/(double x, half y)
	{
       return x / (double)y;
        
    }
    
    inline float operator/(float x, half y)
	{
       return x / (float)y;
        
    }
    
    inline half operator/(unsigned long long x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x / (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator/(long long x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x / (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator/(unsigned long x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x / (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator/(long x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x / (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator/(unsigned int x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x / (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator/(int x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x / (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator/(unsigned short x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x / (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator/(short x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x / (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator/(unsigned char x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x / (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator/(char x, half y)
	{
       double tmp;
       half tmp1;
       tmp = x / (double)y;
       tmp1 = tmp;
       return tmp1;
        
    }

    

    
    inline double operator/( half x, double y)
	{
    
       return (double)x / y;
        
    }
    
    inline float operator/(half x, float y)
	{
       return (float)x / y;
        
    }
    
    inline half operator/( half x, unsigned long long y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x / y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator/(half x, long long y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x / y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator/(half x, unsigned long y)
	{
       double tmp;
       half tmp1;
       tmp =(double)x / y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator/(half x, long y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x / y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator/(half x, unsigned int y)
	{
       double tmp;
       half tmp1;
       tmp =(double)x / y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator/(half x, int y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x / y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator/(half x, unsigned short y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x / y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator/(half x, short y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x / y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator/( half x, unsigned char y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x / y;
       tmp1 = tmp;
       return tmp1;
        
    }
    
    inline half operator/( half x, char y)
	{
       double tmp;
       half tmp1;
       tmp = (double)x / y;
       tmp1 = tmp;
       return tmp1;
        
    }
    

#endif



