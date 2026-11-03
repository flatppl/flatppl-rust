module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.minimum %arg0, %arg1 : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare LE, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.constant dense<9> : tensor<i32>
    %5 = stablehlo.select %2, %4, %3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = stablehlo.constant dense<0.5> : tensor<f32>
    %7 = stablehlo.constant dense<1.0> : tensor<f32>
    %8 = stablehlo.constant dense<true> : tensor<i1>
    %9 = stablehlo.and %8, %8 : tensor<i1>
    %10 = stablehlo.select %9, %6, %6 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = stablehlo.subtract %arg0, %7 : tensor<f32>
    %12 = stablehlo.log %10 : tensor<f32>
    %13 = stablehlo.multiply %11, %12 : tensor<f32>
    %14 = stablehlo.subtract %arg1, %7 : tensor<f32>
    %15 = stablehlo.subtract %7, %10 : tensor<f32>
    %16 = stablehlo.log %15 : tensor<f32>
    %17 = stablehlo.multiply %14, %16 : tensor<f32>
    %18 = chlo.lgamma %arg0 : tensor<f32> -> tensor<f32>
    %19 = stablehlo.negate %18 : tensor<f32>
    %20 = chlo.lgamma %arg1 : tensor<f32> -> tensor<f32>
    %21 = stablehlo.negate %20 : tensor<f32>
    %22 = stablehlo.add %arg0, %arg1 : tensor<f32>
    %23 = chlo.lgamma %22 : tensor<f32> -> tensor<f32>
    %24 = stablehlo.add %13, %17 : tensor<f32>
    %25 = stablehlo.add %19, %21 : tensor<f32>
    %26 = stablehlo.add %25, %23 : tensor<f32>
    %27 = stablehlo.add %24, %26 : tensor<f32>
    %28 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %29 = stablehlo.negate %28 : tensor<f32>
    %30 = stablehlo.select %9, %27, %29 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %30, %5 : tensor<f32>, tensor<i32>
  }
}
