module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.add %arg0, %1 : tensor<f32>
    %3 = chlo.lgamma %2 : tensor<f32> -> tensor<f32>
    %4 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %5 = stablehlo.add %4, %1 : tensor<f32>
    %6 = chlo.lgamma %5 : tensor<f32> -> tensor<f32>
    %7 = stablehlo.negate %6 : tensor<f32>
    %8 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<f32>
    %10 = stablehlo.add %9, %1 : tensor<f32>
    %11 = chlo.lgamma %10 : tensor<f32> -> tensor<f32>
    %12 = stablehlo.negate %11 : tensor<f32>
    %13 = stablehlo.add %3, %7 : tensor<f32>
    %14 = stablehlo.add %13, %12 : tensor<f32>
    %15 = stablehlo.log %arg1 : tensor<f32>
    %16 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %17 = stablehlo.multiply %16, %15 : tensor<f32>
    %18 = stablehlo.subtract %1, %arg1 : tensor<f32>
    %19 = stablehlo.log %18 : tensor<f32>
    %20 = stablehlo.multiply %9, %19 : tensor<f32>
    %21 = stablehlo.add %14, %17 : tensor<f32>
    %22 = stablehlo.add %21, %20 : tensor<f32>
    return %22 : tensor<f32>
  }
}
