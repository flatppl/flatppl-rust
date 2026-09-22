module {
  func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<4.5> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare GE, %arg0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.select %2, %arg0, %3 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %5 = stablehlo.compare EQ, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.select %5, %3, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %7 = stablehlo.log %6 : tensor<f32>
    %8 = stablehlo.multiply %4, %7 : tensor<f32>
    %9 = stablehlo.constant dense<-4.5> : tensor<f32>
    %10 = stablehlo.add %4, %3 : tensor<f32>
    %11 = chlo.lgamma %10 : tensor<f32> -> tensor<f32>
    %12 = stablehlo.negate %11 : tensor<f32>
    %13 = stablehlo.add %8, %9 : tensor<f32>
    %14 = stablehlo.add %13, %12 : tensor<f32>
    %15 = stablehlo.compare EQ, %4, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %16 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %17 = stablehlo.negate %16 : tensor<f32>
    %18 = stablehlo.select %15, %9, %17 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %19 = stablehlo.select %5, %18, %14 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %20 = stablehlo.select %2, %19, %17 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %20 : tensor<f32>
  }
}
