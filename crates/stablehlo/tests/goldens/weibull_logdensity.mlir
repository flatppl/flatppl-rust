module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare GT, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.select %2, %0, %3 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %5 = stablehlo.log %arg0 : tensor<f32>
    %6 = stablehlo.log %arg1 : tensor<f32>
    %7 = stablehlo.negate %6 : tensor<f32>
    %8 = stablehlo.divide %4, %arg1 : tensor<f32>
    %9 = stablehlo.log %8 : tensor<f32>
    %10 = stablehlo.constant dense<1.0> : tensor<f32>
    %11 = stablehlo.subtract %arg0, %10 : tensor<f32>
    %12 = stablehlo.multiply %11, %9 : tensor<f32>
    %13 = stablehlo.power %8, %arg0 : tensor<f32>
    %14 = stablehlo.negate %13 : tensor<f32>
    %15 = stablehlo.add %5, %7 : tensor<f32>
    %16 = stablehlo.add %15, %12 : tensor<f32>
    %17 = stablehlo.add %16, %14 : tensor<f32>
    %18 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %19 = stablehlo.negate %18 : tensor<f32>
    %20 = stablehlo.select %2, %17, %19 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %20 : tensor<f32>
  }
}
