module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %2 = stablehlo.add %1, %arg0 : tensor<f32>
    %3 = chlo.lgamma %2 : tensor<f32> -> tensor<f32>
    %4 = chlo.lgamma %arg0 : tensor<f32> -> tensor<f32>
    %5 = stablehlo.negate %4 : tensor<f32>
    %6 = stablehlo.constant dense<1.0> : tensor<f32>
    %7 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %8 = stablehlo.add %7, %6 : tensor<f32>
    %9 = chlo.lgamma %8 : tensor<f32> -> tensor<f32>
    %10 = stablehlo.negate %9 : tensor<f32>
    %11 = stablehlo.add %3, %5 : tensor<f32>
    %12 = stablehlo.add %11, %10 : tensor<f32>
    %13 = stablehlo.log %arg1 : tensor<f32>
    %14 = stablehlo.add %arg1, %6 : tensor<f32>
    %15 = stablehlo.log %14 : tensor<f32>
    %16 = stablehlo.negate %15 : tensor<f32>
    %17 = stablehlo.add %13, %16 : tensor<f32>
    %18 = stablehlo.multiply %arg0, %17 : tensor<f32>
    %19 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %20 = stablehlo.multiply %19, %15 : tensor<f32>
    %21 = stablehlo.negate %20 : tensor<f32>
    %22 = stablehlo.add %12, %18 : tensor<f32>
    %23 = stablehlo.add %22, %21 : tensor<f32>
    return %23 : tensor<f32>
  }
}
