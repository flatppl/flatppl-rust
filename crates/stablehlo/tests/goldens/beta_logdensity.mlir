module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.compare LE, %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.compare LE, %arg1, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.select %1, %1, %2 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.constant dense<9> : tensor<i32>
    %6 = stablehlo.select %3, %5, %4 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %7 = stablehlo.constant dense<0.5> : tensor<f32>
    %8 = stablehlo.constant dense<1.0> : tensor<f32>
    %9 = stablehlo.constant dense<true> : tensor<i1>
    %10 = stablehlo.and %9, %9 : tensor<i1>
    %11 = stablehlo.select %10, %7, %7 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %12 = stablehlo.subtract %arg0, %8 : tensor<f32>
    %13 = stablehlo.log %11 : tensor<f32>
    %14 = stablehlo.multiply %12, %13 : tensor<f32>
    %15 = stablehlo.subtract %arg1, %8 : tensor<f32>
    %16 = stablehlo.subtract %8, %11 : tensor<f32>
    %17 = stablehlo.log %16 : tensor<f32>
    %18 = stablehlo.multiply %15, %17 : tensor<f32>
    %19 = chlo.lgamma %arg0 : tensor<f32> -> tensor<f32>
    %20 = stablehlo.negate %19 : tensor<f32>
    %21 = chlo.lgamma %arg1 : tensor<f32> -> tensor<f32>
    %22 = stablehlo.negate %21 : tensor<f32>
    %23 = stablehlo.add %arg0, %arg1 : tensor<f32>
    %24 = chlo.lgamma %23 : tensor<f32> -> tensor<f32>
    %25 = stablehlo.add %14, %18 : tensor<f32>
    %26 = stablehlo.add %20, %22 : tensor<f32>
    %27 = stablehlo.add %26, %24 : tensor<f32>
    %28 = stablehlo.add %25, %27 : tensor<f32>
    %29 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %30 = stablehlo.negate %29 : tensor<f32>
    %31 = stablehlo.select %10, %28, %30 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %31, %6 : tensor<f32>, tensor<i32>
  }
}
