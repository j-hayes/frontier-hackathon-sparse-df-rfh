#ifndef error_checking_hpp
#define error_checking_hpp

#include <limits>

bool almostEqual(double x,double y,double tolerance,double zeroTolerance){
  //Threshold denominator so we don't divide by zero
  double threshold=std::numeric_limits<double>::min(); //A very small nonzero number!
  double min=std::min(std::abs(x),std::abs(y));
  if(std::abs(min)==0.0){
    return std::abs(x-y)<zeroTolerance;
  }
  return (std::abs(x-y)/std::max(threshold,min))<tolerance;
}


bool values_are_not_same_and_relevant(double calculated_value, double expected_value){

    return (calculated_value >= 1.0e-5 && expected_value >= 1.0e-5) && 
        !almostEqual(calculated_value, expected_value, 1.0e-3, 1.0e-10);

}

#endif // !error_checking_hpp
