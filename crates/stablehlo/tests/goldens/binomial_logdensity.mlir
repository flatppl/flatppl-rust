module {
  func.func @logdensity(%arg0: tensor<i32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.convert %arg0 : (tensor<i32>) -> tensor<f32>
    %3 = stablehlo.add %2, %1 : tensor<f32>
    %4 = chlo.lgamma %3 : tensor<f32> -> tensor<f32>
    %5 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %6 = stablehlo.add %5, %1 : tensor<f32>
    %7 = chlo.lgamma %6 : tensor<f32> -> tensor<f32>
    %8 = stablehlo.negate %7 : tensor<f32>
    %9 = stablehlo.subtract %arg0, %0 : tensor<i32>
    %10 = stablehlo.convert %9 : (tensor<i32>) -> tensor<f32>
    %11 = stablehlo.add %10, %1 : tensor<f32>
    %12 = chlo.lgamma %11 : tensor<f32> -> tensor<f32>
    %13 = stablehlo.negate %12 : tensor<f32>
    %14 = stablehlo.add %4, %8 : tensor<f32>
    %15 = stablehlo.add %14, %13 : tensor<f32>
    %16 = stablehlo.log %arg1 : tensor<f32>
    %17 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %18 = stablehlo.multiply %17, %16 : tensor<f32>
    %19 = stablehlo.subtract %1, %arg1 : tensor<f32>
    %20 = stablehlo.log %19 : tensor<f32>
    %21 = stablehlo.convert %9 : (tensor<i32>) -> tensor<f32>
    %22 = stablehlo.multiply %21, %20 : tensor<f32>
    %23 = stablehlo.add %15, %18 : tensor<f32>
    %24 = stablehlo.add %23, %22 : tensor<f32>
    return %24 : tensor<f32>
  }
}
